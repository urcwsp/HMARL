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
    parser.add_argument("--param_difficulty", type=str, default="total",
                        choices=["easy", "medium", "hard", "total"],
                        help="动力学参数难度级别: easy(固定值), medium(中等范围), hard(双区间), total(全范围) (保持向后兼容)")
    parser.add_argument("--stage1_difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard", "total"],
                        help="第一阶段参数难度级别")
    parser.add_argument("--stage2_difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "total"],
                        help="第二阶段参数难度级别")
    parser.add_argument("--stage3_difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard", "total"],
                        help="第三阶段参数难度级别")
    parser.add_argument("--test_difficulty", type=str, default="total",
                        choices=["easy", "medium", "hard", "total"],
                        help="测试阶段参数难度级别")
    parser.add_argument("--test_levels", type=str, nargs="+", default=["easy", "medium", "hard"],
                        choices=["easy", "medium", "hard", "total"],
                        help="测试级别列表，可指定多个级别（默认测试所有三个级别）")
    parser.add_argument("--test_episodes_per_level", type=int, default=50,
                        help="每个测试级别的回合数（默认50回合）")
    parser.add_argument("--preview_params", action="store_true",
                        help="预览所有参数难度级别配置后退出")
    parser.add_argument("--test_sampling", type=str, default=None,
                        choices=["easy", "medium", "hard", "total"],
                        help="测试指定难度级别的参数采样功能后退出")
    return parser.parse_args()

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

    def log_test_data(self, test_round, avg_reward, level_results=None):
        """记录测试数据 - 保存测试轮次、整体平均奖励和各级别结果"""
        if not hasattr(self, 'test_data'):
            self.test_data = []

        data = {
            'test_round': test_round,
            'avg_reward': avg_reward
        }

        # 添加各级别的测试结果
        if level_results:
            for level, reward in level_results.items():
                data[f'{level}_reward'] = reward

        self.test_data.append(data)

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

            # 保存测试数据
            if hasattr(self, 'test_data') and self.test_data:
                pd.DataFrame(self.test_data).to_excel(writer, sheet_name='TestResults', index=False)

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


# ============ 多样性度量函数 ============

def pca_entropy_normal(joint_actions, n_components=3):
    """
    思路2: 方法A（多元正态假设）- PCA降维后计算多元正态分布熵
    """
    # PCA降维
    mean = torch.mean(joint_actions, dim=0)
    centered = joint_actions - mean
    U, S, V = torch.svd(centered)

    # 取前n_components个主成分
    n_components = min(n_components, joint_actions.shape[0] - 1, joint_actions.shape[1])
    if n_components <= 0:
        return torch.tensor(0.0, device=joint_actions.device)

    projected = torch.mm(centered, V[:, :n_components])

    # 计算协方差矩阵
    cov = torch.cov(projected.T)

    # 多元正态分布熵: H = (k/2) * log(2πe) + (1/2) * log(det(Σ))
    try:
        eigenvals = torch.linalg.eigvals(cov).real
        eigenvals = torch.clamp(eigenvals, min=1e-8)  # 数值稳定性
        log_det = torch.sum(torch.log(eigenvals))
        entropy = 0.5 * n_components * torch.log(2 * torch.pi * torch.e) + 0.5 * log_det
        return entropy
    except:
        return torch.tensor(0.0, device=joint_actions.device)


def hierarchical_covariance_diversity(joint_actions, state_actions_list):
    """
    思路4: 分层协方差行列式
    joint_actions: [N, 5*action_dim] - N个头的联合动作
    state_actions_list: [[N, action_dim], [N, action_dim], ...] - 每个状态下N个头的动作列表
    """
    # 状态级多样性：每个状态内N个头动作的协方差行列式
    state_diversities = []
    for state_actions in state_actions_list:
        if state_actions.shape[0] > 1:  # 至少需要2个头
            try:
                cov = torch.cov(state_actions.T)
                eigenvals = torch.linalg.eigvals(cov).real
                eigenvals = torch.clamp(eigenvals, min=1e-8)
                log_det = torch.sum(torch.log(eigenvals))
                state_diversities.append(log_det)
            except:
                state_diversities.append(torch.tensor(0.0, device=joint_actions.device))

    # 联合级多样性：N个联合动作的协方差行列式
    try:
        joint_cov = torch.cov(joint_actions.T)
        joint_eigenvals = torch.linalg.eigvals(joint_cov).real
        joint_eigenvals = torch.clamp(joint_eigenvals, min=1e-8)
        joint_log_det = torch.sum(torch.log(joint_eigenvals))
    except:
        joint_log_det = torch.tensor(0.0, device=joint_actions.device)

    # 组合多样性
    if state_diversities:
        state_diversity_mean = torch.mean(torch.stack(state_diversities))
        total_diversity = joint_log_det + state_diversity_mean
    else:
        total_diversity = joint_log_det

    return total_diversity


def spectral_diversity(joint_actions):
    """
    思路3: 谱多样性（奇异值乘积）
    """
    try:
        U, singular_values, V = torch.svd(joint_actions)
        # 使用对数形式避免数值溢出: log(∏ σ_i) = ∑ log(σ_i)
        log_diversity = torch.sum(torch.log(singular_values + 1e-8))
        return log_diversity
    except:
        return torch.tensor(0.0, device=joint_actions.device)


def dpp_diversity(joint_actions, kernel_type='rbf', sigma=None):
    """
    思路3: DPP多样性
    """
    N = joint_actions.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=joint_actions.device)

    try:
        if kernel_type == 'rbf':
            # 自适应sigma
            if sigma is None:
                dists = torch.cdist(joint_actions, joint_actions)
                sigma = torch.median(dists[dists > 0])
                if sigma == 0:
                    sigma = 1.0

            dists_sq = torch.cdist(joint_actions, joint_actions) ** 2
            K = torch.exp(-dists_sq / (2 * sigma ** 2))

        elif kernel_type == 'polynomial':
            inner_products = torch.mm(joint_actions, joint_actions.T)
            K = torch.pow(1 + inner_products, 2)

        else:  # laplacian
            if sigma is None:
                sigma = 1.0
            dists_l1 = torch.cdist(joint_actions, joint_actions, p=1)
            K = torch.exp(-dists_l1 / sigma)

        # 计算 log(det(K))
        eigenvals = torch.linalg.eigvals(K).real
        eigenvals = torch.clamp(eigenvals, min=1e-8)
        log_det = torch.sum(torch.log(eigenvals))

        return log_det
    except:
        return torch.tensor(0.0, device=joint_actions.device)

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

        # 多样性正则化权重
        self.pca_lambda = 0.05  # PCA熵权重
        self.hierarchical_lambda = 0.3   # 分层协方差权重
        self.spectral_lambda = 0.05  # 谱多样性权重
        self.dpp_lambda = 0.1  # DPP权重
        # 总权重 = 0.2 = 4 * 0.05

        # 多样性度量使用开关
        self.use_pca_entropy = False
        self.use_hierarchical = False
        self.use_spectral = True
        self.use_dpp = False

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

            # 原始PPO损失
            ppo_loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 计算多样性正则化（只在多头情况下计算）
            total_diversity_reg = 0.0
            if self.num_heads > 1:
                # 选择状态批次进行联合动作计算 (B=5)
                batch_size = min(5, old_states.shape[0])
                selected_states = old_states[:batch_size]  # [5, state_dim]

                # 每个头对这批状态采样动作 (K=1)
                joint_actions = []
                state_actions_list = []  # 用于分层协方差计算

                # 为分层协方差准备：每个状态下所有头的动作
                for state_idx in range(batch_size):
                    state = selected_states[state_idx].unsqueeze(0)  # [1, state_dim]
                    state_actions = []
                    for h_idx in range(self.num_heads):
                        action, _ = self.policy.act(state, h_idx)
                        state_actions.append(action.squeeze(0))  # [action_dim]
                    state_actions_list.append(torch.stack(state_actions))  # [N, action_dim]

                # 构建每个头的联合动作
                for h_idx in range(self.num_heads):
                    head_joint_actions = []
                    for state_idx in range(batch_size):
                        head_joint_actions.append(state_actions_list[state_idx][h_idx])
                    joint_action = torch.cat(head_joint_actions, dim=0)  # [5*action_dim]
                    joint_actions.append(joint_action)

                # 组成矩阵 [N, 5*action_dim]
                joint_actions_matrix = torch.stack(joint_actions)

                # 计算四种多样性度量
                if self.use_pca_entropy:
                    pca_div = pca_entropy_normal(joint_actions_matrix)
                    total_diversity_reg += -self.pca_lambda * pca_div  # 负号鼓励多样性

                if self.use_hierarchical:
                    hier_div = hierarchical_covariance_diversity(joint_actions_matrix, state_actions_list)
                    total_diversity_reg += -self.hierarchical_lambda * hier_div

                if self.use_spectral:
                    spec_div = spectral_diversity(joint_actions_matrix)
                    total_diversity_reg += -self.spectral_lambda * spec_div

                if self.use_dpp:
                    dpp_div = dpp_diversity(joint_actions_matrix, kernel_type='rbf')
                    total_diversity_reg += -self.dpp_lambda * dpp_div

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

    def set_diversity_weights(self, pca_lambda=None, hierarchical_lambda=None,
                              spectral_lambda=None, dpp_lambda=None):
        """设置不同多样性度量的权重"""
        if pca_lambda is not None:
            self.pca_lambda = pca_lambda
        if hierarchical_lambda is not None:
            self.hierarchical_lambda = hierarchical_lambda
        if spectral_lambda is not None:
            self.spectral_lambda = spectral_lambda
        if dpp_lambda is not None:
            self.dpp_lambda = dpp_lambda

    def set_diversity_usage(self, use_pca_entropy=None, use_hierarchical=None,
                            use_spectral=None, use_dpp=None):
        """设置使用哪些多样性度量"""
        if use_pca_entropy is not None:
            self.use_pca_entropy = use_pca_entropy
        if use_hierarchical is not None:
            self.use_hierarchical = use_hierarchical
        if use_spectral is not None:
            self.use_spectral = use_spectral
        if use_dpp is not None:
            self.use_dpp = use_dpp

    def get_diversity_info(self):
        """获取当前多样性设置信息"""
        return {
            'weights': {
                'pca_lambda': self.pca_lambda,
                'hierarchical_lambda': self.hierarchical_lambda,
                'spectral_lambda': self.spectral_lambda,
                'dpp_lambda': self.dpp_lambda
            },
            'usage': {
                'use_pca_entropy': self.use_pca_entropy,
                'use_hierarchical': self.use_hierarchical,
                'use_spectral': self.use_spectral,
                'use_dpp': self.use_dpp
            }
        }


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

        loss = self.ppo.update(self.memory, self.head_idx)
        self.memory.clear()
        return loss


# 测试函数 - 每200回合测试一次，支持多级别测试
def run_test_episodes(protagonist, env, test_difficulty="total", num_test_episodes=100, test_levels=None):
    """
    运行测试回合，支持多级别参数测试

    Args:
        protagonist: 主角智能体
        env: 环境
        test_difficulty: 基础测试难度
        num_test_episodes: 每个级别的测试回合数
        test_levels: 测试级别列表，如["easy", "medium", "hard"]，None表示只测试基础难度

    Returns:
        测试结果字典 {difficulty: avg_reward}
    """
    # 保存当前环境配置
    original_config = env.config["random_dynamics"].copy()

    # 如果没有指定测试级别，使用基础测试难度
    if test_levels is None:
        test_levels = [test_difficulty]

    test_results = {}

    for level in test_levels:
        print(f"  测试难度级别: {level}")
        level_rewards = []
        level_config = get_param_config_by_difficulty(level)

        for episode in range(num_test_episodes):
            # 根据当前级别采样参数
            random_params = sample_random_params_from_config(level_config, level)

            # 更新环境配置
            for param_name, random_val in random_params.items():
                env.config["random_dynamics"][param_name] = (random_val, random_val)

            # 重置环境
            state, _ = env.reset()

            episode_reward = 0

            for t in range(env.config["horizon"]):
                # 主角选择动作（仅测试，不收集梯度）
                with torch.no_grad():
                    action, _ = protagonist.select_action(state)

                # 环境交互
                next_state, reward, terminated, truncated, info = env.step(action)

                state = next_state
                episode_reward += reward

                if terminated or truncated:
                    break

            level_rewards.append(episode_reward)

        # 计算当前级别的平均奖励
        level_avg_reward = np.mean(level_rewards)
        test_results[level] = level_avg_reward

        print(f"    {level}难度平均奖励: {level_avg_reward:.2f} ({num_test_episodes}回合)")

    # 恢复原始环境配置
    env.config["random_dynamics"] = original_config

    return test_results


# 参数采样函数
def sample_random_params_from_config(param_config, difficulty="total"):
    """
    从参数配置中随机采样参数

    Args:
        param_config: 参数配置字典
        difficulty: 难度级别 ("easy", "medium", "hard", "total")

    Returns:
        随机参数字典
    """
    if difficulty == "easy":
        # 简单难度：直接返回固定值
        return param_config.copy()

    random_params = {}
    for param_name, value in param_config.items():
        if difficulty == "hard" and isinstance(value, list) and len(value) == 2:
            # 困难难度：从两个区间中随机选择一个，然后在该区间内随机采样
            # 修复：使用索引而不是直接选择元组
            range_idx = np.random.choice(len(value))  # 选择0或1
            selected_range = value[range_idx]
            min_val, max_val = selected_range
            random_val = np.random.uniform(min_val, max_val)
        elif isinstance(value, tuple) and len(value) == 2:
            # 中等/总体难度：在单一区间内随机采样
            min_val, max_val = value
            random_val = np.random.uniform(min_val, max_val)
        else:
            # 固定值或其他情况
            random_val = value

        random_params[param_name] = random_val

    return random_params


def get_param_config_by_difficulty(difficulty="total"):
    """
    根据难度级别获取参数配置

    Args:
        difficulty: 难度级别 ("easy", "medium", "hard", "total")

    Returns:
        对应难度的参数配置
    """
    if difficulty == "easy":
        return {
            "max_engine_force": 2350,
            "max_brake_force": 400,
            "wheel_friction": 1.4,
            "max_steering": 45,
            "mass": 900
        }
    elif difficulty == "medium":
        return {
            "max_engine_force": (2025, 2675),
            "max_brake_force": (350, 450),
            "wheel_friction": (1.05, 1.75),
            "max_steering": (32.5, 57.5),
            "mass": (700, 1100)
        }
    elif difficulty == "hard":
        return {
            "max_engine_force": [(1700, 2025), (2675, 3000)],
            "max_brake_force": [(300, 350), (450, 500)],
            "wheel_friction": [(0.7, 1.05), (1.75, 2.1)],
            "max_steering": [(20, 32.5), (57.5, 70)],
            "mass": [(500, 700), (1100, 1300)]
        }
    else:  # total
        return {
            "max_engine_force": (1700, 3000),
            "max_brake_force": (300, 500),
            "wheel_friction": (0.7, 2.1),
            "max_steering": (20, 70),
            "mass": (500, 1300)
        }


def preview_parameter_difficulty():
    """
    预览所有参数难度级别的配置
    """
    print("=" * 80)
    print("动力学参数难度级别配置预览")
    print("=" * 80)

    difficulties = ["easy", "medium", "hard", "total"]

    for difficulty in difficulties:
        print(f"\n【{difficulty.upper()}】难度级别:")
        print("-" * 40)
        config = get_param_config_by_difficulty(difficulty)

        if difficulty == "easy":
            print("固定参数值:")
            for param, value in config.items():
                print(f"  {param:20}: {value}")
        elif difficulty == "medium":
            print("中等难度参数范围:")
            for param, (min_val, max_val) in config.items():
                print(f"  {param:20}: [{min_val}, {max_val}]")
        elif difficulty == "hard":
            print("困难难度参数范围 (双区间):")
            for param, ranges in config.items():
                range1, range2 = ranges
                print(f"  {param:20}: [{range1[0]}, {range1[1]}] 或 [{range2[0]}, {range2[1]}]")
        else:  # total
            print("总体参数范围:")
            for param, (min_val, max_val) in config.items():
                print(f"  {param:20}: [{min_val}, {max_val}]")

    print("\n" + "=" * 80)
    print("使用说明:")
    print("各阶段独立参数配置:")
    print("  --stage1_difficulty easy    : 第一阶段使用固定参数值")
    print("  --stage2_difficulty medium  : 第二阶段使用中等范围参数")
    print("  --stage3_difficulty easy    : 第三阶段使用固定参数值")
    print("  --test_difficulty total     : 测试阶段使用全部范围参数")
    print("\n向后兼容参数 (所有阶段使用相同配置):")
    print("  --param_difficulty easy     : 所有阶段使用固定参数值")
    print("  --param_difficulty medium   : 所有阶段使用中等范围参数")
    print("  --param_difficulty hard     : 所有阶段使用双区间参数")
    print("  --param_difficulty total    : 所有阶段使用全部范围参数")

    print("\n示例命令:")
    print("  # 渐进式训练：从简单到困难，多级别测试")
    print("  python adversarial_ppo_training_senior.py \\")
    print("    --stage1_difficulty easy \\")
    print("    --stage2_difficulty medium \\")
    print("    --stage3_difficulty hard \\")
    print("    --test_levels easy medium hard \\")
    print("    --test_episodes_per_level 30")

    print("\n  # 均匀中等难度训练，仅测试中等和困难级别")
    print("  python adversarial_ppo_training_senior.py \\")
    print("    --param_difficulty medium \\")
    print("    --test_levels medium hard \\")
    print("    --test_episodes_per_level 100")

    print("\n  # 快速测试：仅测试简单级别")
    print("  python adversarial_ppo_training_senior.py \\")
    print("    --test_levels easy \\")
    print("    --test_episodes_per_level 20")

    print("=" * 80)


def test_parameter_sampling(difficulty="medium", num_samples=5):
    """
    测试参数采样功能，生成示例参数

    Args:
        difficulty: 难度级别
        num_samples: 采样数量
    """
    print(f"\n测试 {difficulty.upper()} 难度级别参数采样:")
    print("-" * 50)

    config = get_param_config_by_difficulty(difficulty)

    for i in range(num_samples):
        sampled_params = sample_random_params_from_config(config, difficulty)
        print(f"\n样本 {i+1}:")
        for param, value in sampled_params.items():
            print(f"  {param:20}: {value:.2f}")

    print("-" * 50)


# 三阶段训练函数
def three_stage_ppo_training(pretrained_model_path=None, skip_stage1=False, skip_stage2=False, initial_alpha=0.9, adv_num=1, use_ltc=False, param_difficulty="total", stage1_difficulty="easy", stage2_difficulty="medium", stage3_difficulty="easy", test_difficulty="total", test_levels=None, test_episodes_per_level=50):
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建日志记录器
    logger = TrainingLogger()

    # 获取各阶段的参数配置
    stage1_config = get_param_config_by_difficulty(stage1_difficulty)
    stage2_config = get_param_config_by_difficulty(stage2_difficulty)
    stage3_config = get_param_config_by_difficulty(stage3_difficulty)
    test_config = get_param_config_by_difficulty(test_difficulty)

    # 为了保持向后兼容性，将param_difficulty作为默认值
    param_config = get_param_config_by_difficulty(param_difficulty)
    param_ranges = param_config

    # 默认参数值 - 用于第一阶段训练
    default_params = stage1_config if stage1_difficulty == "easy" else get_param_config_by_difficulty("easy")

    # 设置默认的测试级别
    if test_levels is None:
        test_levels = ["easy", "medium", "hard"]  # 默认测试所有三个级别

    print("=" * 80)
    print("训练参数配置")
    print("=" * 80)
    print(f"第一阶段参数配置 (stage1_difficulty): {stage1_difficulty}")
    print(f"第二阶段参数配置 (stage2_difficulty): {stage2_difficulty}")
    print(f"第三阶段参数配置 (stage3_difficulty): {stage3_difficulty}")
    print(f"测试阶段参数配置 (test_difficulty): {test_difficulty}")
    print(f"测试级别 (test_levels): {test_levels}")
    print(f"每级别测试回合数 (test_episodes_per_level): {test_episodes_per_level}")
    print("=" * 80)

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

    # 第一阶段环境配置 - 使用stage1配置
    if stage1_difficulty == "easy":
        # 简单难度使用固定值
        for param_name, value in stage1_config.items():
            env_config["random_dynamics"][param_name] = (value, value)
    else:
        # 其他难度使用范围的中值作为默认值
        for param_name, value in stage1_config.items():
            if isinstance(value, tuple):
                mid_value = (value[0] + value[1]) / 2
                env_config["random_dynamics"][param_name] = (mid_value, mid_value)
            elif isinstance(value, list):
                # 困难模式，取第一个区间的中值
                mid_value = (value[0][0] + value[0][1]) / 2
                env_config["random_dynamics"][param_name] = (mid_value, mid_value)
            else:
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
    stage3_epochs = 24000  # 第三阶段训练轮数

    # 是否跳过阶段的标志，如果使用预训练模型
    if pretrained_model_path:
        if skip_stage1:
            print(f"使用预训练模型，跳过第一阶段训练")
            stage1_epochs = 0
        if skip_stage2:
            print(f"使用预训练模型，跳过第二阶段训练")
            stage2_epochs = 0

    total_epochs = stage1_epochs + stage2_epochs + stage3_epochs

    #-------------------------- 第一阶段: 环境参数训练 --------------------------#
    if stage1_epochs > 0:
        print(f"\n开始第一阶段训练: {stage1_difficulty}参数配置")
        print("=" * 50)
        print("轮次 | 主角奖励 | 回合长度")
        print("-" * 30)

        stage1_completed = True
        current_epoch = 1
        test_round = 0  # 测试轮次计数器

        while not stage1_completed:
            # 根据第一阶段的难度配置选择参数
            if stage1_difficulty == "easy":
                # 使用固定参数，已经在环境配置中设置
                current_params = stage1_config
            else:
                # 动态采样参数
                current_params = sample_random_params_from_config(stage1_config, stage1_difficulty)
                # 更新环境配置
                for param_name, param_value in current_params.items():
                    env.config["random_dynamics"][param_name] = (param_value, param_value)

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
            logger.log_stage1_data(current_epoch, current_params, episode_reward, episode_length)

            # 更新主角
            loss = protagonist.update(protagonist_memory)
            if loss is not None:
                logger.log_protagonist_loss(current_epoch, loss)
            protagonist_memory.clear()

            # 输出每轮的奖励
            print(f"{current_epoch:3d} | {episode_reward:8.2f} | {episode_length:5d}")

            # 每200轮进行一次测试
            if current_epoch % 200 == 0:
                test_round += 1
                print(f"\n===== 第一阶段第{test_round}轮测试（第{current_epoch}回合后）=====")
                print("开始测试主角在不同难度参数下的表现...")

                # 运行多级别测试
                test_results = run_test_episodes(
                    protagonist, env,
                    test_difficulty=test_difficulty,
                    num_test_episodes=test_episodes_per_level,
                    test_levels=test_levels
                )

                # 记录测试数据 - 记录所有级别的平均值
                overall_avg = np.mean(list(test_results.values()))
                logger.log_test_data(test_round, overall_avg, test_results)

                print(f"测试结果总览: 整体平均奖励={overall_avg:.2f}")
                for level, reward in test_results.items():
                    print(f"  {level}难度: {reward:.2f}")
                print("=" * 50)

            # 每20轮输出一次详细信息并检查是否完成阶段
            if current_epoch % 20 == 0:
                # 计算可用的回合数，最多使用20轮
                available_episodes = min(20, len(protagonist_rewards))
                # 检查最近可用回合的平均奖励
                last_avg = sum(protagonist_rewards[-available_episodes:]) / available_episodes
                print(f"\n当前动力学参数: {stage1_difficulty}难度")
                if stage1_difficulty == "easy":
                    for k, v in current_params.items():
                        print(f"  {k}: {v:.2f}")
                else:
                    # 显示参数范围信息
                    print(f"  使用{stage1_difficulty}难度参数范围")
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
        print(f"\n开始第二阶段训练: {stage2_difficulty}参数配置")
        print("=" * 50)
        print("轮次 | 主角奖励 | 回合长度")
        print("-" * 30)

        stage2_completed = True
        current_epoch = 1
        test_round = 0  # 测试轮次计数器

        while not stage2_completed:
            # 根据第二阶段的难度级别随机选择参数
            random_params = sample_random_params_from_config(stage2_config, stage2_difficulty)

            # 更新环境配置
            for param_name, random_val in random_params.items():
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

            # 每200轮进行一次测试
            if current_epoch % 200 == 0:
                test_round += 1
                print(f"\n===== 第二阶段第{test_round}轮测试（第{current_epoch}回合后）=====")
                print("开始测试主角在不同难度参数下的表现...")

                # 运行多级别测试
                test_results = run_test_episodes(
                    protagonist, env,
                    test_difficulty=test_difficulty,
                    num_test_episodes=test_episodes_per_level,
                    test_levels=test_levels
                )

                # 记录测试数据 - 记录所有级别的平均值
                overall_avg = np.mean(list(test_results.values()))
                logger.log_test_data(test_round, overall_avg, test_results)

                print(f"测试结果总览: 整体平均奖励={overall_avg:.2f}")
                for level, reward in test_results.items():
                    print(f"  {level}难度: {reward:.2f}")
                print("=" * 50)

            # 每20轮输出一次详细信息并检查是否完成阶段
            if current_epoch % 20 == 0:
                # 计算可用的回合数，最多使用20轮
                available_episodes = min(20, len(protagonist_rewards))
                # 检查最近可用回合的平均奖励
                last_avg = sum(protagonist_rewards[-available_episodes:]) / available_episodes
                print(f"\n当前动力学参数: {stage2_difficulty}难度")
                if stage2_difficulty == "easy":
                    for k, v in random_params.items():
                        print(f"  {k}: {v:.2f}")
                else:
                    print(f"  使用{stage2_difficulty}难度参数范围")
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
    print(f"\n开始第三阶段训练: 对抗训练 ({stage3_difficulty}参数配置)")
    print("=" * 50)
    print(f"初始动作混合系数 alpha: {initial_alpha:.2f}")
    print(f"对手数量: {adv_num}")
    print("轮次 | 对手 | Alpha | 主角奖励 | 对手奖励")
    print("-" * 50)

    stage3_completed = False
    current_epoch = 1
    test_round = 0  # 测试轮次计数器

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

            # 根据第三阶段的难度配置选择参数
            if stage3_difficulty == "easy":
                stage3_params = stage3_config
            else:
                stage3_params = sample_random_params_from_config(stage3_config, stage3_difficulty)

            # 更新环境配置
            for param_name, param_value in stage3_params.items():
                env.config["random_dynamics"][param_name] = (param_value, param_value)

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
            params_with_adv = stage3_params.copy()
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
            adversary_loss = adversary.update()
            if adversary_loss is not None:
                logger.log_adversary_loss(current_epoch * adv_num + adv_idx, adversary_loss)

            # 输出每轮的奖励
            print(f"{current_epoch:3d}-{adv_idx} | {adv_idx:3d} | {current_alpha:.1f} | {episode_reward:8.2f} | {adversary_reward:8.2f}")

        # 每200轮进行一次测试
        if current_epoch % 200 == 0:
            test_round += 1
            print(f"\n===== 第三阶段第{test_round}轮测试（第{current_epoch}回合后）=====")
            print("开始测试主角在不同难度参数下的表现...")

            # 运行多级别测试
            test_results = run_test_episodes(
                protagonist, env,
                test_difficulty=test_difficulty,
                num_test_episodes=test_episodes_per_level,
                test_levels=test_levels
            )

            # 记录测试数据 - 记录所有级别的平均值
            overall_avg = np.mean(list(test_results.values()))
            logger.log_test_data(test_round, overall_avg)

            print(f"测试结果总览: 整体平均奖励={overall_avg:.2f}")
            for level, reward in test_results.items():
                print(f"  {level}难度: {reward:.2f}")
            print("=" * 50)

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


def test_trained_agents(protagonist, adversaries, alpha_list=None, num_episodes=10, log_file=None):
    # 创建环境 - 使用默认参数
    env_config = {
        "num_scenarios": 10,
        "horizon": 1000,
        "use_render": False,  # 关闭渲染，避免图形界面错误
        "random_dynamics": {
            "max_engine_force": 2350,
            "max_brake_force": 400,
            "wheel_friction": 1.4,
            "max_steering": 45,
            "mass": 900
        },
        "log_level": 50  # 减少日志输出
    }

    env = VaryingDynamicsEnv(env_config)
    adv_num = len(adversaries)

    # 如果没有提供alpha列表，则使用默认值0.7
    if alpha_list is None:
        alpha_list = [0.7] * adv_num

    # 创建测试日志记录
    test_data = []

    print("\n===== 测试训练好的模型 =====")
    print(f"对手数量: {adv_num}")
    print("轮次 | 对手 | Alpha | 奖励 | 终点 | 碰撞 | 出界 | 动作差异")
    print("-" * 70)

    for i in range(num_episodes):
        # 对每个对手进行测试
        for adv_idx, adversary in enumerate(adversaries):
            # 获取当前对手的alpha值
            current_alpha = alpha_list[adv_idx]

            # 重置环境
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            total_action_diff = 0  # 记录动作差异

            done = False
            while not done:
                # 主角选择动作
                protagonist_action, _ = protagonist.select_action(state)

                # 当前对手选择动作
                adversary_action = adversary.select_action(state)

                # 计算动作差异
                action_diff = np.mean(np.abs(protagonist_action - adversary_action))
                total_action_diff += action_diff

                # 混合动作
                mixed_action = current_alpha * protagonist_action + (1 - current_alpha) * adversary_action

                # 环境交互
                next_state, reward, terminated, truncated, info = env.step(mixed_action)

                state = next_state
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            # 计算平均动作差异
            avg_action_diff = total_action_diff / episode_length if episode_length > 0 else 0

            # 记录测试数据
            test_data.append({
                'episode': i + 1,
                'adversary_idx': adv_idx,
                'alpha': current_alpha,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'arrive_dest': info.get('arrive_dest', False),
                'crash': info.get('crash', False),
                'out_of_road': info.get('out_of_road', False),
                'avg_action_diff': avg_action_diff
            })

            # 输出回合结果
            print(
                f"{i + 1:3d} | {adv_idx:3d} | {current_alpha:.1f} | {episode_reward:6.2f} | {info.get('arrive_dest', False):5} | "
                f"{info.get('crash', False):5} | {info.get('out_of_road', False):5} | {avg_action_diff:.4f}")

    # 如果提供了日志文件，将测试结果添加到Excel文件中
    if test_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log_file = f"logs/test_results_{timestamp}.xlsx"

        # 创建保存目录
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 将测试数据保存到Excel
        test_df = pd.DataFrame(test_data)
        with pd.ExcelWriter(test_log_file, engine='openpyxl') as writer:
            test_df.to_excel(writer, sheet_name='TestResults', index=False)

        print(f"\n测试结果已保存至: {test_log_file}")

    print("\n测试完成！")
    env.close()

    return test_data
# 测试训练好的模型


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 如果用户选择预览参数配置
    if args.preview_params:
        preview_parameter_difficulty()
        exit(0)

    # 如果用户选择测试参数采样
    if args.test_sampling:
        test_parameter_sampling(args.test_sampling, num_samples=5)
        exit(0)

    # 三阶段训练
    protagonist, adversaries, rewards, model_path = three_stage_ppo_training(
        pretrained_model_path=args.pretrained,
        skip_stage1=args.skip_stage1,
        skip_stage2=args.skip_stage2,
        initial_alpha=args.initial_alpha,
        adv_num=args.adv_num,
        use_ltc=args.use_ltc,
        param_difficulty=args.param_difficulty,
        stage1_difficulty=args.stage1_difficulty,
        stage2_difficulty=args.stage2_difficulty,
        stage3_difficulty=args.stage3_difficulty,
        test_difficulty=args.test_difficulty,
        test_levels=args.test_levels,
        test_episodes_per_level=args.test_episodes_per_level
    )

    # 测试
    #test_trained_agents(protagonist, adversaries, alpha_list=None)