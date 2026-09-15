"""
测试训练好的主角智能体
加载已经保存的模型，在不同环境参数下测试其性能
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import json
import pandas as pd  # 添加pandas库用于Excel处理

from metadrive.envs.varying_dynamics_env import VaryingDynamicsEnv

# 设置matplotlib字体
def set_plot_style():
    """设置matplotlib的绘图风格，使用通用字体"""
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用通用字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

# 复制必要的网络架构以便加载模型
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
    
    def act(self, state):
        action_mean = self.actor(state)
        # 直接使用action_var的平方根作为标准差
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach()

# LTC神经网络架构
class LTCActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device, num_units=16):
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
        self.num_units = num_units
        
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
        """LTC神经元的ODE步进"""
        try:
            v_pre = state
            
            # 记录状态变化，用于调试
            states_history = []
            
            # 处理感知输入
            sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
            sensory_rev_activation = sensory_w_activation * self.sensory_erev
            
            # 检查激活值是否正常
            if torch.isnan(sensory_w_activation).any() or torch.isinf(sensory_w_activation).any():
                print(f"警告: 感知激活出现NaN或Inf值")
                # 记录更多信息以便诊断
                print(f"输入范围: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
                print(f"sensory_mu范围: [{self.sensory_mu.min().item():.4f}, {self.sensory_mu.max().item():.4f}]")
                print(f"sensory_sigma范围: [{self.sensory_sigma.min().item():.4f}, {self.sensory_sigma.max().item():.4f}]")
                
                # 修复激活值
                sensory_w_activation = torch.nan_to_num(sensory_w_activation, nan=0.0, posinf=1.0, neginf=-1.0)
                sensory_rev_activation = torch.nan_to_num(sensory_rev_activation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
            w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
            
            # 记录初始状态
            if v_pre.shape[0] == 1:  # 只记录单样本情况，避免日志过多
                states_history.append(v_pre.detach().cpu().numpy()[0])
            
            # ODE求解器步进
            for t in range(self.ode_solver_unfolds):
                w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
                
                # 检查激活值
                if torch.isnan(w_activation).any() or torch.isinf(w_activation).any():
                    print(f"警告: 步骤 {t} 内部激活出现NaN或Inf值")
                    w_activation = torch.nan_to_num(w_activation, nan=0.0, posinf=1.0, neginf=-1.0)
                
                rev_activation = w_activation * self.erev
                
                w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
                w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
                
                numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
                denominator = self.cm_t + self.gleak + w_denominator
                
                # 避免分母为零或过小
                epsilon = 1e-6
                safe_denominator = denominator + epsilon
                
                # 使用更稳定的更新方式
                v_pre_new = numerator / safe_denominator
                
                # 添加状态变化的阻尼，防止大幅震荡
                damping_factor = 0.9  # 阻尼系数
                v_pre = damping_factor * v_pre + (1 - damping_factor) * v_pre_new
                
                # 检查并处理无效值
                if torch.isnan(v_pre).any() or torch.isinf(v_pre).any():
                    print(f"警告: ODE步骤 {t+1} 检测到NaN或Inf值，正在重置")
                    v_pre = torch.where(torch.isnan(v_pre) | torch.isinf(v_pre), 
                                        torch.zeros_like(v_pre), 
                                        v_pre)
                
                # 限制状态值范围，防止过大或过小
                v_pre = torch.clamp(v_pre, min=-5.0, max=5.0)
                
                # 记录状态历史
                if v_pre.shape[0] == 1:  # 只记录单样本情况
                    states_history.append(v_pre.detach().cpu().numpy()[0])
            
            # 每隔一定步数打印状态统计信息
            if np.random.random() < 0.01 and len(states_history) > 0:  # 1%的概率
                states_arr = np.array(states_history)
                print(f"LTC状态统计: 均值={states_arr.mean():.4f}, 标准差={states_arr.std():.4f}, "
                      f"最小值={states_arr.min():.4f}, 最大值={states_arr.max():.4f}")
            
            return v_pre
            
        except Exception as e:
            print(f"ODE步进错误: {e}")
            print(f"输入形状: {inputs.shape}, 状态形状: {state.shape}")
            # 出现错误时返回零状态
            return torch.zeros_like(state)
    
    def act(self, state):
        """通过LTC层处理状态并生成动作"""
        # 确保状态是二维的，即使只有一个样本
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # 检查输入状态是否正常
        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"警告: 输入状态包含NaN或Inf值，已修复")
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 状态归一化，提高数值稳定性
        state_mean = state.mean(dim=1, keepdim=True)
        state_std = state.std(dim=1, keepdim=True) + 1e-6
        normalized_state = (state - state_mean) / state_std
        
        # 初始化为零的状态张量，用于LTC的初始状态
        batch_size = state.shape[0]
        initial_state = torch.zeros(batch_size, self.num_units).to(self.device)
        
        try:
            # 通过LTC层处理状态
            ltc_features = self._ode_step(normalized_state, initial_state)
            
            # 检查LTC特征是否正常
            if torch.isnan(ltc_features).any() or torch.isinf(ltc_features).any():
                print(f"警告: LTC特征包含NaN或Inf值，已修复")
                ltc_features = torch.nan_to_num(ltc_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 限制特征范围
            ltc_features = torch.clamp(ltc_features, min=-10.0, max=10.0)
            
            # 通过actor层生成动作均值
            action_mean = self.actor(ltc_features)
            
            # 检查动作均值是否正常
            if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
                print(f"警告: 动作均值包含NaN或Inf值，已修复")
                action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=0.0, neginf=0.0)
                action_mean = torch.clamp(action_mean, min=-1.0, max=1.0)
            
            # 使用动作方差创建正态分布
            action_std = torch.sqrt(self.action_var)
            
            # 确保标准差不为零
            if torch.isclose(action_std, torch.zeros_like(action_std)).any():
                print(f"警告: 动作标准差接近零，已修复")
                action_std = torch.clamp(action_std, min=0.01)
            
            dist = Normal(action_mean, action_std)
            
            # 从分布中采样动作
            action = dist.sample()
            
            # 直接限制动作范围，确保在[-1,1]之间
            action = torch.clamp(action, min=-1.0, max=1.0)
            
            # 计算动作的对数概率
            try:
                action_logprob = dist.log_prob(action).sum(dim=-1)
            except Exception as e:
                print(f"计算动作对数概率时出错: {e}")
                action_logprob = torch.zeros(batch_size).to(self.device)
            
            # 偶尔打印动作信息
            if np.random.random() < 0.01:  # 1%的概率
                print(f"动作统计: 均值={action.mean().item():.4f}, 标准差={action.std().item():.4f}, "
                      f"最小值={action.min().item():.4f}, 最大值={action.max().item():.4f}")
            
            return action.detach(), action_logprob.detach()
        except Exception as e:
            print(f"LTC模型动作生成错误: {e}")
            print(f"输入状态形状: {state.shape}")
            
            # 发生错误时返回零动作
            zero_action = torch.zeros(batch_size, self.action_dim).to(self.device)
            zero_logprob = torch.zeros(batch_size).to(self.device)
            
            return zero_action.detach(), zero_logprob.detach()

# 简化的PPO类，只包含加载模型和选择动作的功能
class PPO:
    def __init__(self, state_dim, action_dim, action_std_init, device):
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device).to(device)
    
    def load(self, checkpoint_path):
        """加载保存的模型"""
        if os.path.exists(checkpoint_path):
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"成功加载模型: {checkpoint_path}")
        else:
            print(f"警告: 模型文件不存在 {checkpoint_path}")
    
    def load_from_info(self, protagonist_info):
        """从完整信息字典加载模型"""
        # 检查是否是LTC模型
        if protagonist_info.get('is_ltc_model', False):
            # 创建LTC模型并加载参数
            state_dim = protagonist_info['state_dim']
            action_dim = protagonist_info['action_dim']
            action_std = protagonist_info.get('action_std', 0.6)
            num_units = protagonist_info.get('ltc_config', {}).get('num_units', 16)
            
            # 替换当前策略为LTC模型
            self.policy = LTCActorCritic(
                state_dim, 
                action_dim, 
                action_std_init=action_std, 
                device=self.device,
                num_units=num_units
            ).to(self.device)
            
            print(f"创建LTC模型，神经元数量: {num_units}")
        
        # 加载模型参数
        self.policy.load_state_dict(protagonist_info['policy_state_dict'])
        print("成功从完整信息中加载模型")
    
    def select_action(self, state):
        """根据状态选择动作"""
        with torch.no_grad():
            try:
                # 检查状态是否包含NaN或无限值
                if isinstance(state, np.ndarray) and (np.isnan(state).any() or np.isinf(state).any()):
                    print("警告: 输入状态包含NaN或Inf值，已替换为0")
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                
                state = torch.FloatTensor(state).to(self.device)
                
                # 检查是否为LTC模型，如果是，添加更多调试信息
                is_ltc_model = isinstance(self.policy, LTCActorCritic)
                if is_ltc_model and np.random.random() < 0.01:  # 1%的概率
                    print(f"使用LTC模型处理状态，状态形状: {state.shape}")
                
                # 获取动作
                action, _ = self.policy.act(state)
                
                # 确保动作是正确的形状和类型
                action = action.cpu().numpy()
                
                # 检查并修复动作的形状问题
                if len(action.shape) > 1 and action.shape[0] == 1:
                    # 如果动作是2D的且只有一个样本，则展平为1D
                    action = action.flatten()
                
                # 确保每个动作分量都是标量，避免NaN和无限值
                action = np.clip(action, -1.0, 1.0)
                
                # 检查是否有NaN值，如果有则替换为0
                if np.isnan(action).any():
                    print("警告: 检测到NaN动作值，已替换为0")
                    action = np.nan_to_num(action, nan=0.0)
                
                # 记录动作统计信息
                if is_ltc_model and np.random.random() < 0.01:  # 1%的概率
                    print(f"LTC模型最终动作: {action}")
                    print(f"动作范围: [{np.min(action):.4f}, {np.max(action):.4f}]")
                
                return action
                
            except Exception as e:
                print(f"选择动作时出错: {e}")
                # 返回零动作作为后备
                if isinstance(self.policy, LTCActorCritic):
                    action_dim = self.policy.action_dim
                else:
                    # 尝试从策略网络结构推断动作维度
                    try:
                        for name, param in self.policy.named_parameters():
                            if 'actor' in name and '.weight' in name:
                                action_dim = param.shape[0]
                                break
                        else:
                            action_dim = 2  # 默认值
                    except:
                        action_dim = 2  # 默认值
                
                print(f"返回零动作，维度: {action_dim}")
                return np.zeros(action_dim)

def find_latest_model():
    """查找最新的模型文件"""
    # 首先检查训练摘要文件
    summary_files = sorted(glob.glob("models/training_summary_*.json"), reverse=True)
    if summary_files:
        try:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
                if os.path.exists(summary['protagonist_file']):
                    print(f"找到最新训练摘要: {summary_files[0]}")
                    print(f"训练时间: {summary['training_time']}")
                    print(f"最终平均奖励: {summary['final_avg_reward']}")
                    return summary['protagonist_file']
        except Exception as e:
            print(f"读取训练摘要文件失败: {e}")
    
    # 如果没有摘要文件或摘要文件无效，直接查找模型文件
    model_files = sorted(glob.glob("models/protagonist_full_info_*.pt") + 
                         glob.glob("models/three_stage_protagonist_*.pt"), reverse=True)
    if model_files:
        print(f"找到最新模型文件: {model_files[0]}")
        return model_files[0]
    
    # 如果没有找到任何文件，返回默认路径
    print("未找到模型文件，将使用默认路径")
    return "protagonist_full_info.pt"

def load_protagonist_info(info_path):
    """加载主角完整信息"""
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"文件不存在: {info_path}")
    
    try:
        # 尝试使用默认设置加载
        protagonist_info = torch.load(info_path, map_location='cpu')
    except Exception as e:
        print(f"使用默认设置加载失败，尝试使用weights_only=False: {str(e)}")
        # 添加安全的全局变量
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals(["numpy._core.multiarray.scalar"])
        except ImportError:
            pass
        
        try:
            # 使用weights_only=False加载
            protagonist_info = torch.load(info_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            print(f"使用weights_only=False加载也失败: {str(e2)}")
            print("尝试使用pickle加载模型...")
            try:
                import pickle
                with open(info_path, 'rb') as f:
                    protagonist_info = pickle.load(f)
            except Exception as e3:
                print(f"所有尝试都失败，无法加载模型: {str(e3)}")
                raise ValueError(f"无法加载模型 {info_path}, 请检查模型文件格式")
    
    print(f"成功加载主角信息: {info_path}")
    
    # 确保必要的字段存在
    if 'policy_state_dict' not in protagonist_info:
        raise ValueError(f"模型文件缺少policy_state_dict字段，无法加载模型")
    
    if 'state_dim' not in protagonist_info:
        print("警告: 模型文件缺少state_dim字段，将根据模型结构推断")
        # 尝试从模型结构推断state_dim
        try:
            # 查找第一个线性层的输入维度
            for key, value in protagonist_info['policy_state_dict'].items():
                if 'actor.0.weight' in key:  # MLP的第一层
                    protagonist_info['state_dim'] = value.shape[1]
                    print(f"根据模型结构推断state_dim={protagonist_info['state_dim']}")
                    break
                elif 'sensory_mu' in key:  # LTC模型
                    protagonist_info['state_dim'] = value.shape[0]
                    protagonist_info['is_ltc_model'] = True
                    print(f"根据LTC模型结构推断state_dim={protagonist_info['state_dim']}")
                    break
        except Exception as e:
            print(f"无法推断state_dim: {str(e)}")
            protagonist_info['state_dim'] = 28  # 默认值，MetaDrive常用维度
            print(f"使用默认state_dim={protagonist_info['state_dim']}")
    
    if 'action_dim' not in protagonist_info:
        print("警告: 模型文件缺少action_dim字段，将根据模型结构推断")
        # 尝试从模型结构推断action_dim
        try:
            # 查找最后一个线性层的输出维度
            for key, value in protagonist_info['policy_state_dict'].items():
                if 'actor.2.weight' in key or 'actor.4.weight' in key:  # MLP的最后一层
                    protagonist_info['action_dim'] = value.shape[0]
                    print(f"根据模型结构推断action_dim={protagonist_info['action_dim']}")
                    break
                elif 'actor.weight' in key:  # LTC模型的输出层
                    protagonist_info['action_dim'] = value.shape[0]
                    print(f"根据LTC模型结构推断action_dim={protagonist_info['action_dim']}")
                    break
        except Exception as e:
            print(f"无法推断action_dim: {str(e)}")
            protagonist_info['action_dim'] = 2  # 默认值，MetaDrive常用动作维度(方向盘，油门)
            print(f"使用默认action_dim={protagonist_info['action_dim']}")
    
    # 检查是否是LTC模型
    is_ltc_model = protagonist_info.get('is_ltc_model', False)
    
    # 检查模型字典中是否包含LTC特有的参数
    if not is_ltc_model:
        has_ltc_params = any(param for param in protagonist_info['policy_state_dict'].keys() 
                            if any(name in param for name in ['sensory_mu', 'sensory_sigma', 'cm_t', 'vleak']))
        if has_ltc_params:
            print("检测到LTC特有参数，将模型标记为LTC模型")
            protagonist_info['is_ltc_model'] = True
            is_ltc_model = True
            
            # 如果没有ltc_config，创建默认配置
            if 'ltc_config' not in protagonist_info:
                protagonist_info['ltc_config'] = {
                    'num_units': 16,  # 默认神经元数量
                    'ode_solver_unfolds': 6  # 默认ODE步数
                }
    
    if is_ltc_model:
        print("检测到LTC神经网络模型")
        ltc_config = protagonist_info.get('ltc_config', {})
        if ltc_config:
            print(f"LTC配置: 神经元数量={ltc_config.get('num_units', 16)}, "
                 f"ODE步数={ltc_config.get('ode_solver_unfolds', 6)}")
    
    print(f"模型训练时间: {protagonist_info.get('last_update_time', '未知')}")
    print(f"训练轮数: {protagonist_info.get('training_epochs', '未知')}")
    print(f"最终平均奖励: {np.mean(protagonist_info.get('final_rewards', [0])):.2f}")
    
    return protagonist_info

def test_protagonist(info_paths=None, num_episodes=10, use_render=False, num_param_sets=5, random_params=True, save_results=True, use_random_every_episode=False):
    """
    测试训练好的主角智能体
    
    参数:
        info_paths (list): 主角信息文件路径列表，如果为None则自动查找最新模型
        num_episodes (int): 每组参数下的测试回合数
        use_render (bool): 是否启用渲染
        num_param_sets (int): 要测试的随机参数集数量
        random_params (bool): 是否使用随机参数集
        save_results (bool): 是否保存测试结果
        use_random_every_episode (bool): 是否每一回合都使用随机参数
    """
    # 如果未指定模型路径，查找最新的模型
    if info_paths is None:
        info_paths = [find_latest_model()]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载所有主角信息
    protagonists = []
    protagonist_infos = []
    for info_path in info_paths:
        protagonist_info = load_protagonist_info(info_path)
        protagonist_infos.append(protagonist_info)
        
        state_dim = protagonist_info['state_dim']
        action_dim = protagonist_info['action_dim']
        action_std = protagonist_info.get('action_std', 0.6)
        
        # 创建智能体并加载模型
        protagonist = PPO(state_dim, action_dim, action_std_init=action_std, device=device)
        protagonist.load_from_info(protagonist_info)
        protagonists.append(protagonist)
    
    # 创建环境
    env_config = {
        "num_scenarios": 10,
        "horizon": 1000,
        "use_render": use_render,
        "random_dynamics": {
            "max_engine_force": (1500, 1500),  # 固定默认值
            "max_brake_force": (300, 300),
            "wheel_friction": (1.0, 1.0),
            "max_steering": (40, 40),
            "mass": (1500, 1500)
        },
        "log_level": 50  # 减少日志输出
    }
    
    env = VaryingDynamicsEnv(env_config)
    
    # 定义参数范围
    param_ranges = {
        "max_engine_force": (100, 3000),
        "max_brake_force": (20, 600),
        "wheel_friction": (0.1, 2.5),
        "max_steering": (10, 80),
        "mass": (300, 3000)
    }
    
    # 生成随机参数的函数
    def generate_random_params():
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            params[param_name] = float(np.random.uniform(min_val, max_val))
        return params
    
    # 准备所有参数集
    all_param_sets = []
    
    # 1. 添加预定义的固定参数集
    fixed_param_sets = [
        # 默认中等参数
        {
            "max_engine_force": 1500,
            "max_brake_force": 300,
            "wheel_friction": 1.0,
            "max_steering": 40,
            "mass": 1500
        },
        # 高引擎力，低质量 (快速灵活)
        {
            "max_engine_force": 2500,
            "max_brake_force": 400,
            "wheel_friction": 1.5,
            "max_steering": 60,
            "mass": 800
        },
        # 低引擎力，高质量 (缓慢笨重)
        {
            "max_engine_force": 800,
            "max_brake_force": 200,
            "wheel_friction": 0.8,
            "max_steering": 30,
            "mass": 2500
        },
        # 低摩擦，高转向 (易滑动)
        {
            "max_engine_force": 1800,
            "max_brake_force": 350,
            "wheel_friction": 0.3,
            "max_steering": 70,
            "mass": 1200
        },
        # 高摩擦，低转向 (稳定但转弯受限)
        {
            "max_engine_force": 1800,
            "max_brake_force": 350,
            "wheel_friction": 2.0,
            "max_steering": 20,
            "mass": 1800
        }
    ]
    all_param_sets.extend([("fixed", params) for params in fixed_param_sets])
    
    # 2. 添加随机参数集
    if random_params:
        random_param_sets = [generate_random_params() for _ in range(num_param_sets)]
        all_param_sets.extend([("random", params) for params in random_param_sets])
    
    # 3. 如果启用每回合随机参数，添加一个特殊标记
    if use_random_every_episode:
        all_param_sets.append(("random_every_episode", None))
    
    print("\n===== 测试训练好的主角智能体 =====")
    print(f"测试的主角数量: {len(protagonists)}")
    for i, protagonist_info in enumerate(protagonist_infos):
        model_type = "LTC神经网络" if protagonist_info.get('is_ltc_model', False) else "标准MLP网络"
        print(f"主角 {i+1}: {model_type}")
    print(f"参数集总数: {len(all_param_sets)}")
    print("参数集 | 主角 | 回合 | 奖励 | 状态")
    print("-" * 70)
    
    all_results = []
    
    # 对每组参数进行测试
    for param_idx, (param_type, params) in enumerate(all_param_sets):
        param_results = []
        
        # 对每个主角进行测试
        for prot_idx, protagonist in enumerate(protagonists):
            prot_rewards = []
            success_count = 0
            crash_count = 0
            out_of_road_count = 0
            timeout_count = 0
            
            # 在该参数下测试多个回合
            for i in range(num_episodes):
                # 如果是每回合随机参数模式，生成新的随机参数
                if param_type == "random_every_episode":
                    current_params = generate_random_params()
                    # 更新环境参数
                    for k, v in current_params.items():
                        env.config["random_dynamics"][k] = (v, v)
                else:
                    # 使用当前参数集的参数
                    for k, v in params.items():
                        env.config["random_dynamics"][k] = (v, v)
                
                state, _ = env.reset()
                episode_reward = 0
                
                done = False
                timeout = False
                step_count = 0
                
                while not done:
                    # 智能体选择动作
                    action = protagonist.select_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    
                    state = next_state
                    episode_reward += reward
                    done = terminated or truncated
                    step_count += 1
                    
                    if use_render:
                        time.sleep(0.01)  # 渲染模式下减缓速度
                    
                    if step_count >= env.config["horizon"]:
                        timeout = True
                
                prot_rewards.append(episode_reward)
                
                # 记录结果
                if info.get('arrive_dest', False):
                    success_count += 1
                if info.get('crash', False):
                    crash_count += 1
                if info.get('out_of_road', False):
                    out_of_road_count += 1
                if timeout:
                    timeout_count += 1
                
                # 输出每回合的详细信息
                status = "成功" if info.get('arrive_dest', False) else \
                        "碰撞" if info.get('crash', False) else \
                        "出界" if info.get('out_of_road', False) else \
                        "超时" if timeout else "其他"
                print(f"{param_idx+1:4d} | {prot_idx+1:3d} | {i+1:3d} | {episode_reward:8.2f} | {status}")
            
            # 计算统计数据
            avg_reward = sum(prot_rewards) / num_episodes
            success_rate = success_count / num_episodes * 100
            crash_rate = crash_count / num_episodes * 100
            out_of_road_rate = out_of_road_count / num_episodes * 100
            timeout_rate = timeout_count / num_episodes * 100
            
            # 保存测试结果
            prot_result = {
                'protagonist_idx': prot_idx,
                'param_type': param_type,
                'params': params if param_type != "random_every_episode" else "random_every_episode",
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'crash_rate': crash_rate,
                'out_of_road_rate': out_of_road_rate,
                'timeout_rate': timeout_rate,
                'rewards': prot_rewards,
                'is_ltc_model': protagonist_infos[prot_idx].get('is_ltc_model', False)  # 添加模型类型信息
            }
            param_results.append(prot_result)
        
        # 所有主角在当前参数集下测试完成后，输出比较信息
        print(f"\n参数集 {param_idx+1} ({param_type}) 测试完成，所有主角统计信息:")
        print("=" * 70)
        print("主角 | 模型类型 | 平均奖励 | 成功率 | 碰撞率 | 出界率 | 超时率")
        print("-" * 70)
        
        for prot_result in param_results:
            model_type = "LTC" if prot_result['is_ltc_model'] else "MLP"
            print(f"{prot_result['protagonist_idx']+1:3d} | {model_type:8s} | {prot_result['avg_reward']:8.2f} | "
                  f"{prot_result['success_rate']:6.1f}% | {prot_result['crash_rate']:6.1f}% | "
                  f"{prot_result['out_of_road_rate']:6.1f}% | {prot_result['timeout_rate']:6.1f}%")
        
        if param_type != "random_every_episode":
            print("\n当前参数集配置:")
            for k, v in params.items():
                print(f"  {k}: {v:.2f}")
        else:
            print("\n使用每回合随机参数模式")
        print("=" * 70 + "\n")
        
        all_results.append(param_results)
    
    # 保存测试结果
    if save_results:
        # 确保结果目录存在
        if not os.path.exists('results'):
            os.makedirs('results')
            
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 结果文件名
        results_file = f"results/test_results_{timestamp}.pt"
        excel_file = f"results/test_results_{timestamp}.xlsx"
        
        # 准备可序列化的结果
        serializable_results = {
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_infos': [{
                'training_epochs': info.get('training_epochs', 'unknown'),
                'training_time': info.get('last_update_time', 'unknown'),
                'is_ltc_model': info.get('is_ltc_model', False),
                'ltc_config': info.get('ltc_config', {}) if info.get('is_ltc_model', False) else {}
            } for info in protagonist_infos],
            'detailed_results': []
        }
        
        # 转换详细结果
        for param_results in all_results:
            param_serializable = []
            for result in param_results:
                serializable_result = {
                    'protagonist_idx': result['protagonist_idx'],
                    'param_type': result['param_type'],
                    'params': result['params'],
                    'avg_reward': float(result['avg_reward']),
                    'success_rate': float(result['success_rate']),
                    'crash_rate': float(result['crash_rate']),
                    'out_of_road_rate': float(result['out_of_road_rate']),
                    'timeout_rate': float(result['timeout_rate']),
                    'rewards': [float(r) for r in result['rewards']],
                    'is_ltc_model': result['is_ltc_model']
                }
                param_serializable.append(serializable_result)
            serializable_results['detailed_results'].append(param_serializable)
        
        # 保存序列化后的结果
        torch.save(serializable_results, results_file, pickle_protocol=4)
        print(f"\nTest results saved to: {results_file}")
        
        # 创建Excel文件保存结果
        try:
            # 创建一个Excel写入器
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 写入模型信息
                model_info_df = pd.DataFrame(serializable_results['model_infos'])
                model_info_df.insert(0, 'Model ID', [f"Model {i+1}" for i in range(len(model_info_df))])
                model_info_df.to_excel(writer, sheet_name='Model Info', index=False)
                
                # 为每个参数集创建一个工作表
                for param_idx, param_results in enumerate(all_results):
                    # 获取参数类型和值
                    param_type = param_results[0]['param_type']
                    params = param_results[0]['params']
                    
                    # 创建参数集的数据框
                    data = []
                    for result in param_results:
                        model_type = "LTC" if result['is_ltc_model'] else "MLP"
                        data.append({
                            'Agent ID': result['protagonist_idx'] + 1,
                            'Model Type': model_type,
                            'Avg Reward': result['avg_reward'],
                            'Success Rate(%)': result['success_rate'],
                            'Crash Rate(%)': result['crash_rate'],
                            'Out of Road(%)': result['out_of_road_rate'],
                            'Timeout Rate(%)': result['timeout_rate']
                        })
                    
                    # 创建数据框
                    df = pd.DataFrame(data)
                    
                    # 工作表名称
                    sheet_name = f"ParamSet{param_idx+1}"
                    
                    # 写入数据框
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # 如果是固定参数集，添加参数信息
                    if param_type != "random_every_episode" and isinstance(params, dict):
                        # 创建参数信息数据框
                        param_info = pd.DataFrame([params])
                        param_info.to_excel(writer, sheet_name=sheet_name, startrow=len(data)+3, index=False)
                
                # 创建总结工作表
                summary_data = []
                for param_idx, param_results in enumerate(all_results):
                    # 计算每个参数集下的平均性能
                    ltc_results = [r for r in param_results if r['is_ltc_model']]
                    mlp_results = [r for r in param_results if not r['is_ltc_model']]
                    
                    # LTC模型平均性能
                    if ltc_results:
                        ltc_avg_reward = np.mean([r['avg_reward'] for r in ltc_results])
                        ltc_success_rate = np.mean([r['success_rate'] for r in ltc_results])
                        summary_data.append({
                            'Param Set': f"ParamSet{param_idx+1}",
                            'Model Type': 'LTC',
                            'Avg Reward': ltc_avg_reward,
                            'Success Rate(%)': ltc_success_rate
                        })
                    
                    # MLP模型平均性能
                    if mlp_results:
                        mlp_avg_reward = np.mean([r['avg_reward'] for r in mlp_results])
                        mlp_success_rate = np.mean([r['success_rate'] for r in mlp_results])
                        summary_data.append({
                            'Param Set': f"ParamSet{param_idx+1}",
                            'Model Type': 'MLP',
                            'Avg Reward': mlp_avg_reward,
                            'Success Rate(%)': mlp_success_rate
                        })
                
                # 写入总结数据
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"Excel results saved to: {excel_file}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")
        
        # 设置绘图样式
        set_plot_style()
        
        # 创建结果图表
        plot_test_results(all_results, f"results/test_results_{timestamp}.png")
    
    print("\nTest completed!")
    env.close()
    
    return all_results

def plot_test_results(all_results, save_path=None):
    """绘制测试结果图表"""
    num_params = len(all_results)
    num_protagonists = len(all_results[0])
    
    # 检查是否有LTC模型
    has_ltc_models = any(result['is_ltc_model'] for result in all_results[0])
    has_mlp_models = any(not result['is_ltc_model'] for result in all_results[0])
    
    # 创建更多的子图以便比较不同类型的模型
    if has_ltc_models and has_mlp_models:
        fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 15))
    
    # 奖励图
    param_names = [f"P{i+1}" for i in range(num_params)]
    x = np.arange(len(param_names))
    width = 0.8 / num_protagonists
    
    # 为不同模型类型使用不同的颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    ltc_pattern = '/'  # LTC模型使用斜线填充
    
    for prot_idx in range(num_protagonists):
        rewards = [results[prot_idx]['avg_reward'] for results in all_results]
        is_ltc = all_results[0][prot_idx]['is_ltc_model']
        model_type = "LTC" if is_ltc else "MLP"
        
        bar = axes[0].bar(
            x + prot_idx * width, 
            rewards, 
            width, 
            label=f'Agent {prot_idx+1} ({model_type})',
            color=colors[prot_idx % len(colors)],
            hatch=ltc_pattern if is_ltc else None
        )
    
    axes[0].set_title('Average Rewards Across Different Parameter Sets')
    axes[0].set_xlabel('Parameter Set')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_xticks(x + width * (num_protagonists-1)/2)
    axes[0].set_xticklabels(param_names)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 成功率等指标图
    metrics = ['success_rate', 'crash_rate', 'out_of_road_rate', 'timeout_rate']
    metric_names = ['Success Rate', 'Crash Rate', 'Out of Road Rate', 'Timeout Rate']
    markers = ['o', 's', '^', 'D']
    
    for prot_idx in range(num_protagonists):
        is_ltc = all_results[0][prot_idx]['is_ltc_model']
        model_type = "LTC" if is_ltc else "MLP"
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [results[prot_idx][metric] for results in all_results]
            line_style = '--' if is_ltc else '-'
            axes[1].plot(
                x, values, 
                marker=markers[i % len(markers)], 
                linestyle=line_style,
                color=colors[prot_idx % len(colors)],
                label=f'Agent {prot_idx+1} ({model_type}) {metric_name}'
            )
    
    axes[1].set_title('Performance Metrics Across Different Parameter Sets')
    axes[1].set_xlabel('Parameter Set')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 如果同时有LTC和MLP模型，添加直接比较图
    if has_ltc_models and has_mlp_models:
        # 按模型类型分组
        ltc_indices = [i for i, result in enumerate(all_results[0]) if result['is_ltc_model']]
        mlp_indices = [i for i, result in enumerate(all_results[0]) if not result['is_ltc_model']]
        
        # 计算每种类型的平均性能
        metrics_to_compare = ['avg_reward', 'success_rate']
        metric_labels = ['Average Reward', 'Success Rate (%)']
        
        bar_width = 0.35
        x_compare = np.arange(len(param_names))
        
        for i, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
            ltc_values = []
            mlp_values = []
            
            for param_idx in range(num_params):
                # 计算LTC模型的平均值
                if ltc_indices:
                    ltc_avg = np.mean([all_results[param_idx][idx][metric] for idx in ltc_indices])
                    ltc_values.append(ltc_avg)
                
                # 计算MLP模型的平均值
                if mlp_indices:
                    mlp_avg = np.mean([all_results[param_idx][idx][metric] for idx in mlp_indices])
                    mlp_values.append(mlp_avg)
            
            # 绘制比较柱状图
            if ltc_indices:
                axes[2].bar(
                    x_compare - bar_width/2, 
                    ltc_values, 
                    bar_width, 
                    label=f'LTC Model {label}',
                    color='#1f77b4',
                    hatch=ltc_pattern
                )
            
            if mlp_indices:
                axes[2].bar(
                    x_compare + bar_width/2, 
                    mlp_values, 
                    bar_width, 
                    label=f'MLP Model {label}',
                    color='#ff7f0e'
                )
        
        axes[2].set_title('LTC vs MLP Model Performance Comparison')
        axes[2].set_xlabel('Parameter Set')
        axes[2].set_ylabel('Performance Metric')
        axes[2].set_xticks(x_compare)
        axes[2].set_xticklabels(param_names)
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results chart saved to: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained protagonist agents')
    parser.add_argument('--info', type=str, nargs='+', default=None, help='Protagonist info file paths, default uses latest model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes per parameter set')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--no-save', action='store_true', help='Do not save test results')
    parser.add_argument('--random', action='store_true', help='Use random parameters instead of predefined sets')
    parser.add_argument('--num-param-sets', type=int, default=5, help='Number of random parameter sets')
    parser.add_argument('--random-every-episode', action='store_true', help='Use random parameters for each episode')
    parser.add_argument('--compare-models', action='store_true', help='Compare different model types (e.g. LTC and MLP)')
    
    args = parser.parse_args()
    
    # 如果指定了比较模型，尝试加载不同类型的模型
    if args.compare_models and not args.info:
        print("Looking for comparable models...")
        # 查找所有模型文件
        model_files = sorted(glob.glob("models/protagonist_full_info_*.pt") + 
                             glob.glob("models/three_stage_protagonist_*.pt"), reverse=True)
        
        # 尝试找到至少一个LTC模型和一个MLP模型
        ltc_models = []
        mlp_models = []
        
        for model_file in model_files:
            try:
                info = torch.load(model_file, map_location='cpu')
                if info.get('is_ltc_model', False):
                    ltc_models.append(model_file)
                    print(f"Found LTC model: {model_file}")
                else:
                    mlp_models.append(model_file)
                    print(f"Found MLP model: {model_file}")
                
                # 如果每种类型都至少找到一个，则停止搜索
                if ltc_models and mlp_models:
                    break
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
        
        # 组合模型列表进行比较
        compare_models = []
        if ltc_models:
            compare_models.append(ltc_models[0])
        if mlp_models:
            compare_models.append(mlp_models[0])
        
        if len(compare_models) > 1:
            print(f"Will compare the following models: {compare_models}")
            args.info = compare_models
        else:
            print("Could not find enough different model types to compare, will use default model")
    
    test_protagonist(
        info_paths=args.info,
        num_episodes=args.episodes,
        use_render=args.render,
        num_param_sets=args.num_param_sets,
        random_params=args.random,
        save_results=not args.no_save,
        use_random_every_episode=args.random_every_episode
    ) 