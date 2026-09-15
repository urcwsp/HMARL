"""
对抗式PPO算法 - 使用VaryingDynamicsEnv环境
主角：安全高效地驾驶车辆到达目的地
对手：调整环境动力学参数使主角的奖励变少
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from datetime import datetime
import pandas as pd
import json
import argparse

from metadrive.envs.varying_dynamics_env import VaryingDynamicsEnv

# 命令行参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description="对抗式PPO训练 - 可加载预训练模型")
    parser.add_argument("--pretrained", type=str, default=None, 
                        help="预训练主角模型的路径 (.pt文件)")
    parser.add_argument("--skip_stage1", action="store_true", 
                        help="使用预训练模型时是否跳过第一阶段训练")
    parser.add_argument("--skip_stage2", action="store_true", 
                        help="使用预训练模型时是否跳过第二阶段训练")
    parser.add_argument("--adv_num", type=int, default=1,
                        help="对手数量，主角将轮流与每个对手交互")
    parser.add_argument("--initial_alpha", type=float, default=0.9,
                        help="每个对手的初始动作混合系数")
    parser.add_argument("--use_ltc", action="store_true",
                        help="使用液态时间常数(LTC)神经网络代替标准神经网络")
    return parser.parse_args()

# Wasserstein距离计算函数
def wasserstein_2_normal(mean1, std1, mean2, std2):
    """
    计算两个多维正态分布之间的Wasserstein-2距离
    对于N(μ1,σ1²)和N(μ2,σ2²): W2² = ||μ1-μ2||² + ||σ1-σ2||²
    """
    mean_diff = torch.norm(mean1 - mean2, p=2, dim=-1) ** 2
    std_diff = torch.norm(std1 - std2, p=2, dim=-1) ** 2
    return torch.sqrt(mean_diff + std_diff)

# Gower多样性距离计算函数
def gower_distance_normal(mean1, std1, mean2, std2):
    """
    计算两个多维正态分布之间的Gower多样性距离
    Gower距离考虑了不同特征的相对重要性
    对于正态分布，我们使用均值和标准差作为特征
    """
    # 标准化均值差异（使用两个分布标准差的平均值进行标准化）
    avg_std = (std1 + std2) / 2
    mean_diff_normalized = torch.abs(mean1 - mean2) / (avg_std + 1e-8)

    # 标准化标准差差异（使用较大的标准差进行标准化）
    max_std = torch.max(std1, std2)
    std_diff_normalized = torch.abs(std1 - std2) / (max_std + 1e-8)

    # Gower距离是标准化差异的平均值
    gower_dist = torch.mean(mean_diff_normalized + std_diff_normalized, dim=-1)

    return gower_dist

# Hellinger距离计算函数
def hellinger_distance_normal(mean1, std1, mean2, std2):
    """
    计算两个多维正态分布之间的Hellinger距离
    对于两个多维正态分布N(μ1,Σ1)和N(μ2,Σ2)，Hellinger距离为：
    H² = 1 - √(2σ1σ2/(σ1²+σ2²)) * exp(-1/4 * (μ1-μ2)²/(σ1²+σ2²))
    这里简化为对角协方差矩阵的情况
    """
    # 为了数值稳定性，添加小的epsilon
    eps = 1e-8
    std1_safe = std1 + eps
    std2_safe = std2 + eps

    # 计算每个维度的Hellinger距离
    # 几何平均标准差
    geometric_mean_std = torch.sqrt(std1_safe * std2_safe)
    # 算术平均方差
    arithmetic_mean_var = (std1_safe ** 2 + std2_safe ** 2) / 2

    # 标准化系数
    normalization = 2 * geometric_mean_std / (std1_safe ** 2 + std2_safe ** 2 + eps)

    # 指数项
    mean_diff_sq = (mean1 - mean2) ** 2
    exp_term = torch.exp(-0.25 * mean_diff_sq / (arithmetic_mean_var + eps))

    # 每个维度的Hellinger系数
    hellinger_coeff = torch.sqrt(normalization) * exp_term

    # Hellinger距离（每个维度）
    hellinger_dist_per_dim = torch.sqrt(1 - hellinger_coeff + eps)

    # 多维Hellinger距离（取平均或其他聚合方式）
    hellinger_dist = torch.mean(hellinger_dist_per_dim, dim=-1)

    return hellinger_dist

# 加载预训练的主角模型
def load_pretrained_protagonist(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练模型: {model_path}")
    
    print(f"正在加载预训练模型: {model_path}")
    model_info = torch.load(model_path, map_location=device)
    
    # 提取模型参数
    state_dim = model_info.get('state_dim')
    action_dim = model_info.get('action_dim')
    action_std = model_info.get('action_std', 0.6)
    
    # 创建新的PPO对象
    ppo_config = model_info.get('training_config', {})
    lr = ppo_config.get('lr', 3e-5)
    gamma = ppo_config.get('gamma', 0.99)
    K_epochs = ppo_config.get('K_epochs', 20)
    eps_clip = ppo_config.get('eps_clip', 0.2)
    
    protagonist = PPO(
        state_dim, 
        action_dim, 
        action_std_init=action_std, 
        lr=lr,
        gamma=gamma, 
        K_epochs=K_epochs, 
        eps_clip=eps_clip,
        device=device
    )
    
    # 检查模型类型，是否为LTC模型
    is_ltc_model = model_info.get('is_ltc_model', False)
    
    # 如果预训练模型不是LTC模型，但当前使用LTC模型，则需要重新初始化
    if not is_ltc_model and isinstance(protagonist.policy, LTCActorCritic):
        print("警告: 预训练模型不是LTC模型，但当前使用LTC模型。将重新初始化模型权重。")
    # 如果预训练模型是LTC模型，但当前不使用LTC模型，也需要重新初始化
    elif is_ltc_model and not isinstance(protagonist.policy, LTCActorCritic):
        print("警告: 预训练模型是LTC模型，但当前不使用LTC模型。将重新初始化模型权重。")
    # 只有当模型类型匹配时才加载权重
    elif 'policy_state_dict' in model_info:
        try:
            protagonist.policy.load_state_dict(model_info['policy_state_dict'])
            protagonist.policy_old.load_state_dict(model_info['policy_state_dict'])
            print(f"成功加载预训练模型权重")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            print("将使用随机初始化的模型权重")
    else:
        print("警告: 模型文件中没有找到策略权重!")
    
    return protagonist, model_info

# Excel日志记录类
class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f"{log_dir}/training_log_{self.timestamp}.xlsx"
        
        # 初始化DataFrame用于存储数据
        self.stage1_data = []
        self.stage2_data = []
        self.stage3_data = []
        
        self.protagonist_loss_values = []
        self.adversary_loss_values = []
    
    def log_stage1_data(self, epoch, params, protagonist_reward, episode_length):
        data = {
            'epoch': epoch,
            'episode_length': episode_length,
            'protagonist_reward': protagonist_reward
        }
        # 添加环境参数
        for param_name, value in params.items():
            data[param_name] = value
        
        self.stage1_data.append(data)
    
    def log_stage2_data(self, epoch, params, protagonist_reward, episode_length):
        data = {
            'epoch': epoch,
            'episode_length': episode_length,
            'protagonist_reward': protagonist_reward
        }
        # 添加环境参数
        for param_name, value in params.items():
            data[param_name] = value
        
        self.stage2_data.append(data)
    
    def log_stage3_data(self, epoch, params, protagonist_reward, adversary_reward, episode_length, alpha=0.7, action_diff=None):
        data = {
            'epoch': epoch,
            'episode_length': episode_length,
            'protagonist_reward': protagonist_reward,
            'adversary_reward': adversary_reward,
            'alpha': alpha
        }
        
        # 添加动作差异（如果提供）
        if action_diff is not None:
            data['avg_action_diff'] = action_diff
            
        # 添加环境参数
        for param_name, value in params.items():
            data[param_name] = value
        
        self.stage3_data.append(data)
    
    def log_protagonist_loss(self, epoch, loss_value):
        self.protagonist_loss_values.append({
            'epoch': epoch,
            'loss': float(loss_value.mean().item()) if hasattr(loss_value, 'mean') else float(loss_value)
        })
    
    def log_adversary_loss(self, epoch, loss_value):
        self.adversary_loss_values.append({
            'epoch': epoch,
            'loss': float(loss_value.mean().item()) if hasattr(loss_value, 'mean') else float(loss_value)
        })
    
    def save_to_excel(self):
        # 创建ExcelWriter对象
        with pd.ExcelWriter(self.log_file, engine='openpyxl') as writer:
            # 将每个阶段的数据保存到不同的工作表
            if self.stage1_data:
                pd.DataFrame(self.stage1_data).to_excel(writer, sheet_name='Stage1', index=False)
            if self.stage2_data:
                pd.DataFrame(self.stage2_data).to_excel(writer, sheet_name='Stage2', index=False)
            if self.stage3_data:
                pd.DataFrame(self.stage3_data).to_excel(writer, sheet_name='Stage3', index=False)
            
            # 保存损失值
            if self.protagonist_loss_values:
                pd.DataFrame(self.protagonist_loss_values).to_excel(writer, sheet_name='ProtagonistLoss', index=False)
            if self.adversary_loss_values:
                pd.DataFrame(self.adversary_loss_values).to_excel(writer, sheet_name='AdversaryLoss', index=False)
        
        print(f"训练日志已保存至: {self.log_file}")

# 标准神经网络架构 - 用于主角
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device
        
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        
        # 值函数网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
    
    def set_action_std(self, new_action_std):
        self.action_var = torch.full(self.action_var.shape, new_action_std * new_action_std).to(self.device)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        # 直接使用action_var的平方根作为标准差
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_value = self.critic(state).squeeze()
        
        return action_logprobs, state_value, dist_entropy
    
    def clip_parameters(self):
        """空实现，保持接口一致性"""
        pass


# LTC神经网络架构 - 用于主角
class LTCActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(LTCActorCritic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # LTC参数初始化
        self.ode_solver_unfolds = 6  # ODE求解器步数
        
        # 初始化参数范围
        self.w_init_max = 1.0
        self.w_init_min = 0.01
        self.cm_init_min = 0.5
        self.cm_init_max = 0.5
        self.gleak_init_min = 1
        self.gleak_init_max = 1
        self.erev_init_factor = 1
        
        # 参数约束范围
        self.w_min_value = 0.00001
        self.w_max_value = 1000
        self.gleak_min_value = 0.00001
        self.gleak_max_value = 1000
        self.cm_t_min_value = 0.000001
        self.cm_t_max_value = 1000
        
        # LTC神经元数量 (隐藏层大小)
        self.num_units = 16
        
        # 共享的LTC层参数
        # 感知输入参数
        self.sensory_mu = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(3.0, 8.0))
        self.sensory_W = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(self.w_init_min, self.w_init_max))
        
        # 随机初始化感知输入的平衡电位 (-1 或 1)
        sensory_erev_init = 2 * torch.randint(0, 2, (state_dim, self.num_units)) - 1
        self.sensory_erev = nn.Parameter(sensory_erev_init.float() * self.erev_init_factor)
        
        # 内部连接参数
        self.mu = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(3.0, 8.0))
        self.W = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(self.w_init_min, self.w_init_max))
        
        # 随机初始化内部连接的平衡电位 (-1 或 1)
        erev_init = 2 * torch.randint(0, 2, (self.num_units, self.num_units)) - 1
        self.erev = nn.Parameter(erev_init.float() * self.erev_init_factor)
        
        # 神经元参数
        self.vleak = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(-0.2, 0.2))
        self.gleak = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(self.gleak_init_min, self.gleak_init_max))
        self.cm_t = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(self.cm_init_min, self.cm_init_max))
        
        # 策略网络输出层
        self.actor = nn.Linear(self.num_units, action_dim)
        
        # 值函数网络输出层
        self.critic = nn.Linear(self.num_units, 1)
        
        # 动作方差
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
    
    def _sigmoid(self, v_pre, mu, sigma):
        """计算sigmoid激活函数"""
        # 重塑v_pre以便进行广播
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)
    
    def _ode_step(self, inputs, state):
        """半隐式欧拉方法求解ODE"""
        v_pre = state
        
        # 计算感知输入的激活
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        
        # 计算感知输入的分子和分母
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        
        # 多次迭代求解ODE
        for t in range(self.ode_solver_unfolds):
            # 计算内部连接的激活
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            
            # 计算总的分子和分母
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            
            # 计算新的状态
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator
            
            v_pre = numerator / denominator
        
        return v_pre
    
    def set_action_std(self, new_action_std):
        self.action_var = torch.full(self.action_var.shape, new_action_std * new_action_std).to(self.device)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # 通过LTC层处理状态
        # 确保状态是二维的，即使只有一个样本
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))
        
        # 通过actor层生成动作均值
        action_mean = self.actor(ltc_features)
        
        # 使用动作方差创建正态分布
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        
        # 从分布中采样动作
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        # 通过LTC层处理状态
        # 确保状态是二维的，即使只有一个样本
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))
        
        # 通过actor层生成动作均值
        action_mean = self.actor(ltc_features)
        
        # 使用动作方差创建正态分布
        action_var = self.action_var.expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))
        
        # 计算动作的对数概率
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        # 通过critic层计算状态值
        state_value = self.critic(ltc_features).squeeze()
        
        return action_logprobs, state_value, dist_entropy
    
    def clip_parameters(self):
        """约束参数在有效范围内"""
        with torch.no_grad():
            # 约束权重
            self.W.clamp_(self.w_min_value, self.w_max_value)
            self.sensory_W.clamp_(self.w_min_value, self.w_max_value)
            
            # 约束神经元参数
            self.gleak.clamp_(self.gleak_min_value, self.gleak_max_value)
            self.cm_t.clamp_(self.cm_t_min_value, self.cm_t_max_value)


# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim, action_std_init, lr, gamma, K_epochs, eps_clip, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.policy = LTCActorCritic(state_dim, action_dim, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = LTCActorCritic(state_dim, action_dim, action_std_init, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        
        # 确保返回的是标量而不是数组，如果是单样本的情况
        if action.shape[0] == 1:
            return action.cpu().numpy().flatten(), action_logprob
        else:
            return action.cpu().numpy(), action_logprob
    
    def update(self, memory):
        # 检查记忆缓冲区是否为空
        if len(memory.rewards) == 0:
            print("警告: 记忆缓冲区为空，跳过更新")
            return None
            
        # 蒙特卡洛估计回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 标准化奖励，确保奖励列表至少有两个元素
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 转换为张量
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        # 初始化最终损失值
        final_loss = 0
        
        # 多次优化
        for _ in range(self.K_epochs):
            # 评估旧动作和状态
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # 重要性采样比率
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算优势函数
            advantages = rewards - state_values.detach()
            
            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 确保rewards和state_values具有相同的维度
            # 修复：确保state_values调整为和rewards相同的形状
            if rewards.shape != state_values.shape:
                if len(rewards.shape) == 1 and len(state_values.shape) == 1:
                    # 如果都是一维，但长度不同，可能是批处理大小不同
                    min_length = min(rewards.shape[0], state_values.shape[0])
                    rewards = rewards[:min_length]
                    state_values = state_values[:min_length]
                else:
                    # 确保两者都是一维或二维
                    rewards = rewards.view(-1)
                    state_values = state_values.view(-1)
                    # 再次检查长度
                    min_length = min(rewards.shape[0], state_values.shape[0])
                    rewards = rewards[:min_length]
                    state_values = state_values[:min_length]
            
            # 最终损失 = 策略损失 + 值函数损失 - 熵正则化
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 保存最后一次迭代的损失
            if _ == self.K_epochs - 1:
                final_loss = loss.mean()
            
            # 梯度优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            # 对LTC模型参数进行裁剪，确保参数在有效范围内
            self.policy.clip_parameters()
        
        # 复制新权重到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 返回最终损失值用于日志记录
        return final_loss


# 经验回放缓冲区
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# 标准多头对手网络架构
class MultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, num_heads, device):
        super(MultiHeadActorCritic, self).__init__()
        self.device = device
        self.num_heads = num_heads
        
        # 共享的前两层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # 多头策略网络 - 每个对手一个头
        self.actor_heads = nn.ModuleList([
            nn.Linear(64, action_dim) for _ in range(num_heads)
        ])
        
        # 多头值函数网络 - 每个对手一个头
        self.critic_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_heads)
        ])
        
        # 为每个头创建动作方差
        self.action_vars = [torch.full((action_dim,), action_std_init * action_std_init).to(device) 
                           for _ in range(num_heads)]
        
    def set_action_std(self, new_action_std, head_idx=None):
        if head_idx is not None:
            # 设置特定头的动作标准差
            self.action_vars[head_idx] = torch.full(self.action_vars[head_idx].shape, 
                                                   new_action_std * new_action_std).to(self.device)
        else:
            # 设置所有头的动作标准差
            for i in range(self.num_heads):
                self.action_vars[i] = torch.full(self.action_vars[i].shape, 
                                                new_action_std * new_action_std).to(self.device)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, head_idx):
        # 通过共享层
        shared_features = self.shared_layers(state)
        
        # 通过特定头的actor层
        action_mean = self.actor_heads[head_idx](shared_features)
        
        # 使用对应头的action_var
        dist = Normal(action_mean, torch.sqrt(self.action_vars[head_idx]))
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action, head_idx):
        # 通过共享层
        shared_features = self.shared_layers(state)
        
        # 通过特定头的actor层
        action_mean = self.actor_heads[head_idx](shared_features)
        
        # 使用对应头的action_var
        action_var = self.action_vars[head_idx].expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        # 通过特定头的critic层
        state_value = self.critic_heads[head_idx](shared_features).squeeze()
        
        return action_logprobs, state_value, dist_entropy
    
    def clip_parameters(self):
        """空实现，保持接口一致性"""
        pass

    def get_all_heads_output(self, state):
        """获取所有头的输出用于计算Wasserstein距离"""
        # 通过共享层
        shared_features = self.shared_layers(state)

        # 获取所有头的输出
        all_action_means = []
        all_action_stds = []

        for head_idx in range(self.num_heads):
            action_mean = self.actor_heads[head_idx](shared_features)
            action_std = torch.sqrt(self.action_vars[head_idx]).expand_as(action_mean)
            all_action_means.append(action_mean)
            all_action_stds.append(action_std)

        return all_action_means, all_action_stds



# 液态时间常数多头对手网络架构
class LTCMultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, num_heads, device):
        super(LTCMultiHeadActorCritic, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # LTC参数初始化
        self.ode_solver_unfolds = 6  # ODE求解器步数
        
        # 初始化参数范围
        self.w_init_max = 1.0
        self.w_init_min = 0.01
        self.cm_init_min = 0.5
        self.cm_init_max = 0.5
        self.gleak_init_min = 1
        self.gleak_init_max = 1
        self.erev_init_factor = 1
        
        # 参数约束范围
        self.w_min_value = 0.00001
        self.w_max_value = 1000
        self.gleak_min_value = 0.00001
        self.gleak_max_value = 1000
        self.cm_t_min_value = 0.000001
        self.cm_t_max_value = 1000
        
        # LTC神经元数量 (隐藏层大小)
        self.num_units = 16
        
        # 共享的LTC层参数
        # 感知输入参数
        self.sensory_mu = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(3.0, 8.0))
        self.sensory_W = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(self.w_init_min, self.w_init_max))
        
        # 随机初始化感知输入的平衡电位 (-1 或 1)
        sensory_erev_init = 2 * torch.randint(0, 2, (state_dim, self.num_units)) - 1
        self.sensory_erev = nn.Parameter(sensory_erev_init.float() * self.erev_init_factor)
        
        # 内部连接参数
        self.mu = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(3.0, 8.0))
        self.W = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(self.w_init_min, self.w_init_max))
        
        # 随机初始化内部连接的平衡电位 (-1 或 1)
        erev_init = 2 * torch.randint(0, 2, (self.num_units, self.num_units)) - 1
        self.erev = nn.Parameter(erev_init.float() * self.erev_init_factor)
        
        # 神经元参数
        self.vleak = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(-0.2, 0.2))
        self.gleak = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(self.gleak_init_min, self.gleak_init_max))
        self.cm_t = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(self.cm_init_min, self.cm_init_max))
        
        # 多头策略网络 - 每个对手一个头
        self.actor_heads = nn.ModuleList([
            nn.Linear(self.num_units, action_dim) for _ in range(num_heads)
        ])
        
        # 多头值函数网络 - 每个对手一个头
        self.critic_heads = nn.ModuleList([
            nn.Linear(self.num_units, 1) for _ in range(num_heads)
        ])
        
        # 为每个头创建动作方差
        self.action_vars = [torch.full((action_dim,), action_std_init * action_std_init).to(device) 
                           for _ in range(num_heads)]
    
    def _sigmoid(self, v_pre, mu, sigma):
        """计算sigmoid激活函数"""
        # 重塑v_pre以便进行广播
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)
    
    def _ode_step(self, inputs, state):
        """半隐式欧拉方法求解ODE"""
        v_pre = state
        
        # 计算感知输入的激活
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        
        # 计算感知输入的分子和分母
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        
        # 多次迭代求解ODE
        for t in range(self.ode_solver_unfolds):
            # 计算内部连接的激活
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            
            # 计算总的分子和分母
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            
            # 计算新的状态
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator
            
            v_pre = numerator / denominator
        
        return v_pre
    
    def set_action_std(self, new_action_std, head_idx=None):
        if head_idx is not None:
            # 设置特定头的动作标准差
            self.action_vars[head_idx] = torch.full(self.action_vars[head_idx].shape, 
                                                   new_action_std * new_action_std).to(self.device)
        else:
            # 设置所有头的动作标准差
            for i in range(self.num_heads):
                self.action_vars[i] = torch.full(self.action_vars[i].shape, 
                                                new_action_std * new_action_std).to(self.device)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, head_idx):
        # 通过LTC层处理状态
        # 确保状态是二维的，即使只有一个样本
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))
        
        # 通过特定头的actor层
        action_mean = self.actor_heads[head_idx](ltc_features)
        
        # 使用对应头的action_var
        dist = Normal(action_mean, torch.sqrt(self.action_vars[head_idx]))
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action, head_idx):
        # 通过LTC层处理状态
        # 确保状态是二维的，即使只有一个样本
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))
        
        # 通过特定头的actor层
        action_mean = self.actor_heads[head_idx](ltc_features)
        
        # 使用对应头的action_var
        action_var = self.action_vars[head_idx].expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))
        
        # 计算动作的对数概率
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        # 通过特定头的critic层
        state_value = self.critic_heads[head_idx](ltc_features).squeeze()
        
        return action_logprobs, state_value, dist_entropy
    
    def clip_parameters(self):
        """约束参数在有效范围内"""
        with torch.no_grad():
            # 约束权重
            self.W.clamp_(self.w_min_value, self.w_max_value)
            self.sensory_W.clamp_(self.w_min_value, self.w_max_value)
            
            # 约束神经元参数
            self.gleak.clamp_(self.gleak_min_value, self.gleak_max_value)
            self.cm_t.clamp_(self.cm_t_min_value, self.cm_t_max_value)

    def get_all_heads_output(self, state):
        """获取所有头的输出用于计算Wasserstein距离"""
        # 通过LTC层处理状态
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))

        # 获取所有头的输出
        all_action_means = []
        all_action_stds = []

        for head_idx in range(self.num_heads):
            action_mean = self.actor_heads[head_idx](ltc_features)
            action_std = torch.sqrt(self.action_vars[head_idx]).expand_as(action_mean)
            all_action_means.append(action_mean)
            all_action_stds.append(action_std)

        return all_action_means, all_action_stds



# 多头对手PPO算法
class MultiHeadPPO:
    def __init__(self, state_dim, action_dim, action_std_init, lr, gamma, K_epochs, eps_clip, num_heads, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.num_heads = num_heads
        
        self.policy = LTCMultiHeadActorCritic(state_dim, action_dim, action_std_init, num_heads, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = LTCMultiHeadActorCritic(state_dim, action_dim, action_std_init, num_heads, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        # 多种距离正则化权重
        self.wasserstein_lambda = 0.2  # Wasserstein距离权重
        self.gower_lambda = 0.15  # Gower距离权重
        self.hellinger_lambda = 0.1  # Hellinger距离权重

        # 距离度量选择（可以通过参数控制使用哪些距离）
        self.use_wasserstein = True
        self.use_gower = False
        self.use_hellinger = False

    def set_distance_weights(self, wasserstein_lambda=None, gower_lambda=None, hellinger_lambda=None):
        """设置不同距离度量的权重"""
        if wasserstein_lambda is not None:
            self.wasserstein_lambda = wasserstein_lambda
        if gower_lambda is not None:
            self.gower_lambda = gower_lambda
        if hellinger_lambda is not None:
            self.hellinger_lambda = hellinger_lambda

    def set_distance_usage(self, use_wasserstein=True, use_gower=True, use_hellinger=True):
        """设置使用哪些距离度量"""
        self.use_wasserstein = use_wasserstein
        self.use_gower = use_gower
        self.use_hellinger = use_hellinger

    def select_action(self, state, head_idx):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state, head_idx)
        
        # 确保返回的是标量而不是数组，如果是单样本的情况
        if action.shape[0] == 1:
            return action.cpu().numpy().flatten(), action_logprob
        else:
            return action.cpu().numpy(), action_logprob
    
    def update(self, memory, head_idx):
        # 检查记忆缓冲区是否为空
        if len(memory.rewards) == 0:
            print(f"警告: 对手{head_idx}的记忆缓冲区为空，跳过更新")
            return None
            
        # 蒙特卡洛估计回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 标准化奖励，确保奖励列表至少有两个元素
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 转换为张量
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        # 初始化最终损失值
        final_loss = 0
        
        # 多次优化
        for _ in range(self.K_epochs):
            # 评估旧动作和状态，使用特定的头
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, head_idx)
            
            # 重要性采样比率
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算优势函数
            advantages = rewards - state_values.detach()
            
            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 确保rewards和state_values具有相同的维度
            if rewards.shape != state_values.shape:
                if len(rewards.shape) == 1 and len(state_values.shape) == 1:
                    # 如果都是一维，但长度不同，可能是批处理大小不同
                    min_length = min(rewards.shape[0], state_values.shape[0])
                    rewards = rewards[:min_length]
                    state_values = state_values[:min_length]
                else:
                    # 确保两者都是一维或二维
                    rewards = rewards.view(-1)
                    state_values = state_values.view(-1)
                    # 再次检查长度
                    min_length = min(rewards.shape[0], state_values.shape[0])
                    rewards = rewards[:min_length]
                    state_values = state_values[:min_length]
            
            # 最终损失 = 策略损失 + 值函数损失 - 熵正则化
            # 原始PPO损失
            ppo_loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 计算多种距离正则化项
            total_diversity_reg = 0.0
            distance_info = {}  # 用于记录各种距离的信息

            if self.num_heads > 1:  # 只有多头时才计算
                # 获取所有头的输出
                all_action_means, all_action_stds = self.policy.get_all_heads_output(old_states)

                # 计算当前头与其他头之间的各种距离
                current_mean = all_action_means[head_idx]
                current_std = all_action_stds[head_idx]

                # 存储不同距离的值
                wasserstein_distances = []
                gower_distances = []
                hellinger_distances = []

                for other_idx in range(self.num_heads):
                    if other_idx != head_idx:
                        other_mean = all_action_means[other_idx]
                        other_std = all_action_stds[other_idx]

                        # 计算Wasserstein距离
                        if self.use_wasserstein:
                            w_dist = wasserstein_2_normal(current_mean, current_std, other_mean, other_std)
                            wasserstein_distances.append(w_dist)

                        # 计算Gower距离
                        if self.use_gower:
                            g_dist = gower_distance_normal(current_mean, current_std, other_mean, other_std)
                            gower_distances.append(g_dist)

                        # 计算Hellinger距离
                        if self.use_hellinger:
                            h_dist = hellinger_distance_normal(current_mean, current_std, other_mean, other_std)
                            hellinger_distances.append(h_dist)

                # 组合不同的距离正则化项
                if wasserstein_distances and self.use_wasserstein:
                    wasserstein_reg = -torch.mean(torch.stack(wasserstein_distances))  # 负号表示鼓励多样性
                    total_diversity_reg += self.wasserstein_lambda * wasserstein_reg
                    distance_info['wasserstein'] = torch.mean(torch.stack(wasserstein_distances)).item()

                if gower_distances and self.use_gower:
                    gower_reg = -torch.mean(torch.stack(gower_distances))  # 负号表示鼓励多样性
                    total_diversity_reg += self.gower_lambda * gower_reg
                    distance_info['gower'] = torch.mean(torch.stack(gower_distances)).item()

                if hellinger_distances and self.use_hellinger:
                    hellinger_reg = -torch.mean(torch.stack(hellinger_distances))  # 负号表示鼓励多样性
                    total_diversity_reg += self.hellinger_lambda * hellinger_reg
                    distance_info['hellinger'] = torch.mean(torch.stack(hellinger_distances)).item()

            # 最终损失 = PPO损失 + 多样性正则化
            loss = ppo_loss + total_diversity_reg
            
            # 保存最后一次迭代的损失
            if _ == self.K_epochs - 1:
                final_loss = loss.mean()
            
            # 梯度优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            # 对LTC模型参数进行裁剪，确保参数在有效范围内
            self.policy.clip_parameters()
        
        # 复制新权重到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 返回最终损失值用于日志记录
        return final_loss


# 对手智能体 - 生成对抗动作
class AdversaryAgent:
    def __init__(self, device, lr=3e-5):
        self.device = device
        
        # 对手输入状态维度与主角相同
        self.state_dim = None  # 将在训练函数中设置
        self.action_dim = None  # 将在训练函数中设置
        
        # 初始化对手的PPO算法
        self.ppo = None  # 将在训练函数中初始化
        self.head_idx = None  # 将在训练函数中设置
        
        self.memory = RolloutBuffer()
        self.learning_rate = lr
    
    def initialize(self, state_dim, action_dim, head_idx=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.head_idx = head_idx
        
        # 注意：多头PPO将在外部初始化，这里只存储引用
    
    def set_ppo(self, ppo):
        self.ppo = ppo
    
    def select_action(self, state):
        if self.ppo is None:
            raise ValueError("对手智能体尚未初始化")
            
        action, logprob = self.ppo.select_action(state, self.head_idx)
        self.memory.states.append(torch.tensor(state, dtype=torch.float32).to(self.device))
        self.memory.actions.append(torch.tensor(action, dtype=torch.float32).to(self.device))
        self.memory.logprobs.append(logprob)
        
        return action

    def update(self):
        if self.ppo is None:
            raise ValueError("对手智能体尚未初始化")

        result = self.ppo.update(self.memory, self.head_idx)
        self.memory.clear()

        # 处理返回值 - 可能包含距离信息
        if isinstance(result, tuple):
            loss, distance_info = result
            return loss, distance_info
        else:
            return result, None


# 三阶段训练函数
def three_stage_ppo_training(pretrained_model_path=None, skip_stage1=False, skip_stage2=False, initial_alpha=0.9, adv_num=1, use_ltc=False):
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建日志记录器
    logger = TrainingLogger()
    
    # 可调整的动力学参数范围
    param_ranges = {
        "max_engine_force": (1700, 3000),
        "max_brake_force": (300, 500),
        "wheel_friction": (0.7, 2.1),
        "max_steering": (20, 70),
        "mass": (500, 1300)
    }
    
    # 默认参数值 - 用于第一阶段训练
    default_params = {
        "max_engine_force": 2350,
        "max_brake_force": 400,
        "wheel_friction": 1.4,
        "max_steering": 45,
        "mass": 900
    }
    
    # 创建环境配置
    env_config = {
        "num_scenarios": 100,  # 场景数量
        "horizon": 1000,       # 每个episode的最大步数
        "random_dynamics": {}, # 稍后填充
        "show_logo": False,     # 关闭logo显示
        "show_interface": False, # 关闭界面显示
        "debug": False,         # 关闭调试信息
        "log_level": 50         # 只显示错误级别的日志 (50 = CRITICAL)
    }
    
    # 第一阶段环境配置 - 使用默认参数
    for param_name, value in default_params.items():
        env_config["random_dynamics"][param_name] = (value, value)
    
    # 创建环境
    env = VaryingDynamicsEnv(env_config)
    
    # 主角智能体设置
    protagonist_state_dim = env.observation_space.shape[0]
    protagonist_action_dim = env.action_space.shape[0]
    
    # 创建多头对手PPO算法
    if use_ltc:
        print("对手使用液态时间常数(LTC)神经网络架构")
        multi_head_ppo = MultiHeadPPO(
            protagonist_state_dim,
            protagonist_action_dim,
            action_std_init=0.6,
            lr=3e-5,
            gamma=0.99,
            K_epochs=10,
            eps_clip=0.2,
            num_heads=adv_num,
            device=device
        )
    else:
        print("对手使用标准神经网络架构")
        # 临时保存原始的LTCMultiHeadActorCritic类引用
        OriginalMultiHeadActorCritic = globals()['LTCMultiHeadActorCritic']
        # 替换为标准MultiHeadActorCritic类
        globals()['LTCMultiHeadActorCritic'] = MultiHeadActorCritic
        
        multi_head_ppo = MultiHeadPPO(
            protagonist_state_dim,
            protagonist_action_dim,
            action_std_init=0.6,
            lr=3e-5,
            gamma=0.99,
            K_epochs=10,
            eps_clip=0.2,
            num_heads=adv_num,
            device=device
        )
        
        # 恢复原始类引用
        globals()['LTCMultiHeadActorCritic'] = OriginalMultiHeadActorCritic
    
    # 创建多个对手智能体
    adversaries = [AdversaryAgent(device, lr=3e-5) for _ in range(adv_num)]
    
    # 为每个对手初始化alpha值和奖励历史
    adversary_alphas = [initial_alpha for _ in range(adv_num)]
    adversary_active = [True for _ in range(adv_num)]  # 标记每个对手是否仍然活跃
    
    # 初始化所有对手智能体
    for i, adversary in enumerate(adversaries):
        adversary.initialize(protagonist_state_dim, protagonist_action_dim, head_idx=i)
        adversary.set_ppo(multi_head_ppo)
    
    # 记录rewards历史
    protagonist_rewards = []
    adversary_rewards = [[] for _ in range(adv_num)]  # 为每个对手创建奖励列表
    
    # 加载预训练模型或创建新模型
    model_info = None
    if pretrained_model_path:
        protagonist, model_info = load_pretrained_protagonist(pretrained_model_path, device)
        # 验证模型与环境维度一致
        if isinstance(protagonist.policy, LTCActorCritic):
            # 检查LTC模型的输入输出维度
            if protagonist.policy.state_dim != protagonist_state_dim or \
               protagonist.policy.action_dim != protagonist_action_dim:
                raise ValueError(f"预训练LTC模型维度与环境不匹配! 模型: {protagonist.policy.state_dim}->{protagonist.policy.action_dim}, 环境: {protagonist_state_dim}->{protagonist_action_dim}")
        else:
            # 检查标准模型的输入输出维度
            if protagonist.policy.actor[0].in_features != protagonist_state_dim or \
               protagonist.policy.actor[-1].out_features != protagonist_action_dim:
                raise ValueError(f"预训练标准模型维度与环境不匹配! 模型: {protagonist.policy.actor[0].in_features}->{protagonist.policy.actor[-1].out_features}, 环境: {protagonist_state_dim}->{protagonist_action_dim}")
        
        # 从模型信息中提取奖励历史（如果有）
        if model_info and 'rewards_history' in model_info:
            print(f"从预训练模型加载了 {len(model_info['rewards_history'])} 个历史奖励记录")
    else:
        # 创建新的主角智能体，根据use_ltc参数选择网络架构
        if use_ltc:
            print("使用液态时间常数(LTC)神经网络架构")
            # 创建LTC主角智能体
            protagonist = PPO(
                protagonist_state_dim, 
                protagonist_action_dim, 
                action_std_init=0.6, 
                lr=3e-5,
                gamma=0.99, 
                K_epochs=20, 
                eps_clip=0.2,
                device=device
            )
        else:
            # 使用标准网络架构
            # 临时保存原始的LTCActorCritic类引用
            OriginalActorCritic = globals()['LTCActorCritic']
            # 替换为标准ActorCritic类
            globals()['LTCActorCritic'] = ActorCritic
            
            protagonist = PPO(
                protagonist_state_dim, 
                protagonist_action_dim, 
                action_std_init=0.6, 
                lr=3e-5,
                gamma=0.99, 
                K_epochs=20, 
                eps_clip=0.2,
                device=device
            )
            
            # 恢复原始类引用
            globals()['LTCActorCritic'] = OriginalActorCritic
    
    protagonist_memory = RolloutBuffer()
    
    # 训练参数
    stage1_epochs = 300  # 第一阶段训练轮数
    stage2_epochs = 300  # 第二阶段训练轮数
    stage3_epochs = 500000  # 第三阶段训练轮数
    
    # 是否跳过阶段的标志，如果使用预训练模型
    if pretrained_model_path:
        if skip_stage1:
            print(f"使用预训练模型，跳过第一阶段训练")
            stage1_epochs = 0
        if skip_stage2:
            print(f"使用预训练模型，跳过第二阶段训练")
            stage2_epochs = 0
    
    total_epochs = stage1_epochs + stage2_epochs + stage3_epochs
    
    #-------------------------- 第一阶段: 默认环境参数训练 --------------------------#
    if stage1_epochs > 0:
        print("\n开始第一阶段训练: 默认环境参数")
        print("=" * 50)
        print("轮次 | 主角奖励 | 回合长度")
        print("-" * 30)
        
        stage1_completed = True
        current_epoch = 1
        
        while not stage1_completed:
            # 重置环境
            state, _ = env.reset()
            
            # 单回合交互
            episode_reward = 0
            episode_length = 0
            
            for t in range(env.config["horizon"]):
                # 主角选择动作
                action, logprob = protagonist.select_action(state)
                
                # 环境交互
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # 保存到主角的回放缓冲区
                protagonist_memory.states.append(torch.tensor(state, dtype=torch.float32).to(device))
                protagonist_memory.actions.append(torch.tensor(action, dtype=torch.float32).to(device))
                protagonist_memory.logprobs.append(logprob)
                protagonist_memory.rewards.append(reward)
                protagonist_memory.is_terminals.append(terminated or truncated)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # 记录主角奖励
            protagonist_rewards.append(episode_reward)
            
            # 记录本轮数据到日志
            logger.log_stage1_data(current_epoch, default_params, episode_reward, episode_length)
            
            # 更新主角
            loss = protagonist.update(protagonist_memory)
            if loss is not None:
                logger.log_protagonist_loss(current_epoch, loss)
            protagonist_memory.clear()
            
            # 输出每轮的奖励
            print(f"{current_epoch:3d} | {episode_reward:8.2f} | {episode_length:5d}")
            
            # 每20轮输出一次详细信息并检查是否完成阶段
            if current_epoch % 20 == 0:
                # 计算可用的回合数，最多使用20轮
                available_episodes = min(20, len(protagonist_rewards))
                # 检查最近可用回合的平均奖励
                last_avg = sum(protagonist_rewards[-available_episodes:]) / available_episodes
                print(f"\n当前动力学参数: 默认参数")
                for k, v in default_params.items():
                    print(f"  {k}: {v:.2f}")
                print(f"主角累积奖励: {last_avg:.2f} ({available_episodes}轮平均)")
                
                # 检查是否达到进入下一阶段的条件
                if last_avg > 1200 and current_epoch >= stage1_epochs and available_episodes >= 20:
                    stage1_completed = True
                    print(f"第一阶段完成! {available_episodes}轮平均奖励已超过1300")
                elif current_epoch >= stage1_epochs:
                    print(f"已完成最小轮数{stage1_epochs}，但平均奖励{last_avg:.2f}小于1300，继续训练")
                
                print("-" * 50 + "\n")
            
            current_epoch += 1
    else:
        print("\n跳过第一阶段训练")
    
    #-------------------------- 第二阶段: 随机环境参数训练 --------------------------#
    if stage2_epochs > 0:
        print("\n开始第二阶段训练: 随机环境参数")
        print("=" * 50)
        print("轮次 | 主角奖励 | 回合长度")
        print("-" * 30)
        
        stage2_completed = True
        current_epoch = 1
        
        while not stage2_completed:
            # 随机选择参数
            random_params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                random_val = np.random.uniform(min_val, max_val)
                random_params[param_name] = random_val
                # 更新环境配置
                env.config["random_dynamics"][param_name] = (random_val, random_val)
            
            # 重置环境 - 应用新的随机参数
            state, _ = env.reset()
            
            # 单回合交互
            episode_reward = 0
            episode_length = 0
            
            for t in range(env.config["horizon"]):
                # 主角选择动作
                action, logprob = protagonist.select_action(state)
                
                # 环境交互
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # 保存到主角的回放缓冲区
                protagonist_memory.states.append(torch.tensor(state, dtype=torch.float32).to(device))
                protagonist_memory.actions.append(torch.tensor(action, dtype=torch.float32).to(device))
                protagonist_memory.logprobs.append(logprob)
                protagonist_memory.rewards.append(reward)
                protagonist_memory.is_terminals.append(terminated or truncated)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # 记录主角奖励
            protagonist_rewards.append(episode_reward)
            
            # 记录本轮数据到日志
            logger.log_stage2_data(current_epoch, random_params, episode_reward, episode_length)
            
            # 更新主角
            loss = protagonist.update(protagonist_memory)
            if loss is not None:
                logger.log_protagonist_loss(current_epoch + stage1_epochs, loss)
            protagonist_memory.clear()
            
            # 输出每轮的奖励
            print(f"{current_epoch:3d} | {episode_reward:8.2f} | {episode_length:5d}")
            
            # 每20轮输出一次详细信息并检查是否完成阶段
            if current_epoch % 20 == 0:
                # 计算可用的回合数，最多使用20轮
                available_episodes = min(20, len(protagonist_rewards))
                # 检查最近可用回合的平均奖励
                last_avg = sum(protagonist_rewards[-available_episodes:]) / available_episodes
                print(f"\n当前动力学参数: 随机参数")
                for k, v in random_params.items():
                    print(f"  {k}: {v:.2f}")
                print(f"主角累积奖励: {last_avg:.2f} ({available_episodes}轮平均)")
                
                # 检查是否达到进入下一阶段的条件
                if last_avg > 1200 and current_epoch >= stage2_epochs and available_episodes >= 20:
                    stage2_completed = True
                    print(f"第二阶段完成! {available_episodes}轮平均奖励已超过1300")
                elif current_epoch >= stage2_epochs:
                    print(f"已完成最小轮数{stage2_epochs}，但平均奖励{last_avg:.2f}小于1300，继续训练")
                
                print("-" * 50 + "\n")
            
            current_epoch += 1
    else:
        print("\n跳过第二阶段训练")
    
    #-------------------------- 第三阶段: 对抗训练 --------------------------#
    print("\n开始第三阶段训练: 对抗训练")
    print("=" * 50)
    print(f"初始动作混合系数 alpha: {initial_alpha:.2f}")
    print(f"对手数量: {adv_num}")
    print("轮次 | 对手 | Alpha | 主角奖励 | 对手奖励")
    print("-" * 50)
    
    stage3_completed = False
    current_epoch = 1
    
    while not stage3_completed:
        # 检查是否所有对手都已经不活跃（alpha <= 0.5）
        if not any(adversary_active):
            print("\n所有对手的alpha值都已降至0.5或以下，训练结束!")
            stage3_completed = True
            break
            
        # 每个epoch轮流与每个活跃的对手交互一次
        for adv_idx, adversary in enumerate(adversaries):
            # 如果当前对手不活跃（alpha <= 0.5），则跳过
            if not adversary_active[adv_idx]:
                continue
                
            # 获取当前对手的alpha值
            current_alpha = adversary_alphas[adv_idx]
            
            # 重置环境 - 使用默认参数
            for param_name, value in default_params.items():
                env.config["random_dynamics"][param_name] = (value, value)
            
            state, _ = env.reset()
            
            # 单回合交互
            episode_reward = 0
            episode_length = 0
            total_action_diff = 0  # 跟踪动作差异
            
            for t in range(env.config["horizon"]):
                # 主角选择动作
                protagonist_action, logprob = protagonist.select_action(state)
                
                # 当前对手选择动作
                adversary_action = adversary.select_action(state)
                
                # 计算动作差异
                action_diff = np.mean(np.abs(protagonist_action - adversary_action))
                total_action_diff += action_diff
                
                # 混合动作：alpha * 主角动作 + (1-alpha) * 对手动作
                mixed_action = current_alpha * protagonist_action + (1-current_alpha) * adversary_action
                
                # 环境交互
                next_state, reward, terminated, truncated, info = env.step(mixed_action)
                
                # 保存到主角的回放缓冲区
                protagonist_memory.states.append(torch.tensor(state, dtype=torch.float32).to(device))
                protagonist_memory.actions.append(torch.tensor(protagonist_action, dtype=torch.float32).to(device))
                protagonist_memory.logprobs.append(logprob)
                protagonist_memory.rewards.append(reward)
                protagonist_memory.is_terminals.append(terminated or truncated)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # 计算平均动作差异
            avg_action_diff = total_action_diff / episode_length if episode_length > 0 else 0
            
            # 记录主角奖励
            protagonist_rewards.append(episode_reward)
            
            # 对手奖励是主角奖励的相反数
            adversary_reward = -episode_reward
            adversary_rewards[adv_idx].append(adversary_reward)
            
            # 记录本轮数据到日志，添加对手索引信息和alpha值
            params_with_adv = default_params.copy()
            params_with_adv['adversary_idx'] = adv_idx
            params_with_adv['alpha'] = current_alpha
            logger.log_stage3_data(current_epoch, params_with_adv, episode_reward, adversary_reward, 
                                episode_length, alpha=current_alpha, action_diff=avg_action_diff)
            
            # 更新对手的奖励和终止信号
            adversary.memory.rewards.append(adversary_reward)
            adversary.memory.is_terminals.append(True)
            
            # 每轮都更新主角
            protagonist_loss = protagonist.update(protagonist_memory)
            if protagonist_loss is not None:
                logger.log_protagonist_loss(current_epoch + stage1_epochs + stage2_epochs, protagonist_loss)
            protagonist_memory.clear()
            
            # 每轮都更新当前对手
            adversary_result = adversary.update()
            if adversary_result[0] is not None:
                adversary_loss, distance_info = adversary_result
                logger.log_adversary_loss(current_epoch * adv_num + adv_idx, adversary_loss)
            
            # 输出每轮的奖励
            print(f"{current_epoch:3d}-{adv_idx} | {adv_idx:3d} | {current_alpha:.1f} | {episode_reward:8.2f} | {adversary_reward:8.2f}")
        
        # 每20轮检查是否需要调整alpha值，在所有对手都交互完后进行
        if current_epoch % 20 == 0:
            # 计算可用的回合数，最多使用20轮
            available_episodes = min(20, len(protagonist_rewards))
            # 检查最近可用回合的平均奖励
            last_protagonist_avg = sum(protagonist_rewards[-available_episodes:]) / available_episodes
            
            print(f"\n当前训练轮次: {current_epoch}")
            print(f"回合长度: {episode_length}")
            print(f"主角累积奖励: {last_protagonist_avg:.2f} ({available_episodes}轮平均)")
            
            # 输出每个对手的平均奖励和alpha值，并检查是否需要调整alpha
            print("\n对手状态:")
            for adv_idx, adv_rewards in enumerate(adversary_rewards):
                if adv_rewards:  # 确保有奖励记录
                    adv_available_episodes = min(20, len(adv_rewards))
                    last_adv_avg = sum(adv_rewards[-adv_available_episodes:]) / adv_available_episodes
                    status = "活跃" if adversary_active[adv_idx] else "不活跃"
                    print(f"对手{adv_idx}: alpha={adversary_alphas[adv_idx]:.1f}, 奖励={last_adv_avg:.2f} ({adv_available_episodes}轮平均), 状态={status}")
                    
                    # 检查是否需要调整alpha值
                    if adversary_active[adv_idx] and last_adv_avg < -1200:
                        adversary_alphas[adv_idx] -= 0.1
                        print(f"对手{adv_idx}的平均奖励小于-1200，alpha值减少0.1，现在为: {adversary_alphas[adv_idx]:.1f}")
                        
                        # 如果alpha值降至0.5或以下，标记该对手为不活跃
                        if adversary_alphas[adv_idx] <= 0.5:
                            adversary_active[adv_idx] = False
                            print(f"对手{adv_idx}的alpha值已降至{adversary_alphas[adv_idx]:.1f}，不再与该对手交互")
            
            # 检查是否达到完成训练的条件
            if not any(adversary_active):
                stage3_completed = True
                print("\n所有对手的alpha值都已降至0.5或以下，训练结束!")
            elif current_epoch >= stage3_epochs:
                stage3_completed = True
                print(f"已完成最大轮数{stage3_epochs}，训练结束")

            print("-" * 50 + "\n")
        
        current_epoch += 1
    
    print("三阶段训练完成!")
    
    # 保存训练日志到Excel文件
    logger.save_to_excel()
    
    # 计算最终平均奖励，确保使用正确的回合数
    final_episodes = min(20, len(protagonist_rewards))
    final_avg_reward = float(np.mean(protagonist_rewards[-final_episodes:]))
    
    # 合并之前的奖励历史（如果有）
    if model_info and 'rewards_history' in model_info:
        previous_rewards = model_info['rewards_history']
        all_rewards = previous_rewards + [float(r) for r in protagonist_rewards]
    else:
        all_rewards = [float(r) for r in protagonist_rewards]
    
    # 保存主角的完整信息，包括模型参数、网络结构、状态维度、动作维度等
    protagonist_info = {
        'state_dim': int(protagonist_state_dim),
        'action_dim': int(protagonist_action_dim),
        'action_std': 0.6,  # 初始标准差
        'policy_state_dict': protagonist.policy.state_dict(),
        'is_ltc_model': isinstance(protagonist.policy, LTCActorCritic),  # 标记是否为LTC模型
        'training_config': {
            'lr': 3e-5,
            'gamma': 0.99,
            'K_epochs': 20,
            'eps_clip': 0.2,
            'initial_alpha': initial_alpha,  # 添加初始alpha参数
            'adv_num': adv_num,  # 添加对手数量参数
            'final_alphas': adversary_alphas,  # 添加最终alpha值
        },
        'rewards_history': all_rewards,  # 所有奖励历史
        'final_rewards': [float(r) for r in protagonist_rewards[-final_episodes:]],  # 最后可用轮数的奖励
        'final_episodes': final_episodes,
        'training_epochs': int(total_epochs),
        'stage_info': {
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs,
            'stage3_epochs': stage3_epochs
        },
        'last_update_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pretrained_model': pretrained_model_path,  # 记录是否使用了预训练模型
        'initial_alpha': initial_alpha,  # 添加初始alpha值
        'final_alphas': adversary_alphas,  # 添加最终alpha值列表
        'adv_num': adv_num,  # 添加对手数量
        'use_ltc': use_ltc  # 添加是否使用LTC模型
    }
    
    # 如果是LTC模型，添加LTC特定参数
    if isinstance(protagonist.policy, LTCActorCritic):
        protagonist_info['ltc_config'] = {
            'ode_solver_unfolds': protagonist.policy.ode_solver_unfolds,
            'num_units': protagonist.policy.num_units,
            'w_init_range': [protagonist.policy.w_init_min, protagonist.policy.w_init_max],
            'cm_init_range': [protagonist.policy.cm_init_min, protagonist.policy.cm_init_max],
            'gleak_init_range': [protagonist.policy.gleak_init_min, protagonist.policy.gleak_init_max],
            'erev_init_factor': protagonist.policy.erev_init_factor
        }
    
    # 创建保存目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存文件名添加时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    protagonist_file = f"models/three_stage_protagonist_{timestamp}.pt"
    
    # 保存所有对手模型
    adversary_files = []
    
    # 保存多头PPO模型
    multi_head_ppo_file = f"models/multi_head_adversary_ppo_{timestamp}.pth"
    multi_head_ppo_info = {
        'policy_state_dict': multi_head_ppo.policy.state_dict(),
        'num_heads': adv_num,
        'alphas': adversary_alphas,
        'active': adversary_active,
        'is_ltc_model': isinstance(multi_head_ppo.policy, LTCMultiHeadActorCritic),  # 标记是否为LTC模型
        'use_ltc': use_ltc  # 添加是否使用LTC模型
    }
    
    # 如果是LTC模型，添加LTC特定参数
    if isinstance(multi_head_ppo.policy, LTCMultiHeadActorCritic):
        multi_head_ppo_info['ltc_config'] = {
            'ode_solver_unfolds': multi_head_ppo.policy.ode_solver_unfolds,
            'num_units': multi_head_ppo.policy.num_units,
            'w_init_range': [multi_head_ppo.policy.w_init_min, multi_head_ppo.policy.w_init_max],
            'cm_init_range': [multi_head_ppo.policy.cm_init_min, multi_head_ppo.policy.cm_init_max],
            'gleak_init_range': [multi_head_ppo.policy.gleak_init_min, multi_head_ppo.policy.gleak_init_max],
            'erev_init_factor': multi_head_ppo.policy.erev_init_factor
        }
    
    torch.save(multi_head_ppo_info, multi_head_ppo_file)
    print(f"已保存多头对手PPO模型至: {multi_head_ppo_file}")
    
    # 记录每个对手的信息（主要是索引）
    for adv_idx, adversary in enumerate(adversaries):
        adversary_file = f"models/three_stage_adversary_{adv_idx}_{timestamp}.json"
        adversary_info = {
            'head_idx': adv_idx,
            'alpha': adversary_alphas[adv_idx],
            'active': adversary_active[adv_idx],
            'multi_head_ppo_file': multi_head_ppo_file
        }
        with open(adversary_file, 'w') as f:
            json.dump(adversary_info, f, indent=4)
        adversary_files.append(adversary_file)
        print(f"已保存对手{adv_idx}信息至: {adversary_file}")
    
    # 保存为单个文件
    torch.save(protagonist_info, protagonist_file, pickle_protocol=4)
    print(f"已保存主角完整信息至: {protagonist_file}")
    
    # 创建简单的训练信息文件
    info_summary = {
        'timestamp': timestamp,
        'protagonist_file': protagonist_file,
        'adversary_files': adversary_files,
        'multi_head_ppo_file': multi_head_ppo_file,
        'training_epochs': total_epochs,
        'stage_info': {
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs,
            'stage3_epochs': stage3_epochs
        },
        'final_avg_reward': final_avg_reward,
        'training_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'excel_log_file': logger.log_file,  # 添加Excel日志文件路径
        'pretrained_model': pretrained_model_path,  # 记录是否使用了预训练模型
        'initial_alpha': initial_alpha,  # 添加初始alpha值
        'final_alphas': adversary_alphas,  # 添加最终alpha值列表
        'adv_num': adv_num,  # 添加对手数量
        'use_ltc': use_ltc  # 添加是否使用LTC模型
    }
    
    # 保存训练信息摘要
    summary_file = f"models/three_stage_training_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(info_summary, f, indent=4)
    print(f"已保存训练摘要至: {summary_file}")
    
    # 关闭环境
    env.close()
    
    return protagonist, adversaries, protagonist_rewards, protagonist_file


# 测试训练好的模型


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 三阶段训练
    protagonist, adversaries, rewards, model_path = three_stage_ppo_training(
        pretrained_model_path=args.pretrained,
        skip_stage1=args.skip_stage1,
        skip_stage2=args.skip_stage2,
        initial_alpha=args.initial_alpha,
        adv_num=args.adv_num,
        use_ltc=args.use_ltc
    )
    
    # 测试
    #test_trained_agents(protagonist, adversaries, alpha_list=None)