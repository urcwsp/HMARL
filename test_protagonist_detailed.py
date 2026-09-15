"""
测试训练好的主角模型 - 输出详细的环境信息
包括：速度、加速度、车头时距等
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import pandas as pd
from datetime import datetime
import argparse

from metadrive.envs.varying_dynamics_env import VaryingDynamicsEnv


# 复制必要的网络架构类
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

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
        pass


class LTCActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(LTCActorCritic, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ode_solver_unfolds = 6
        self.w_init_max = 1.0
        self.w_init_min = 0.01
        self.cm_init_min = 0.5
        self.cm_init_max = 0.5
        self.gleak_init_min = 1
        self.gleak_init_max = 1
        self.erev_init_factor = 1

        self.w_min_value = 0.00001
        self.w_max_value = 1000
        self.gleak_min_value = 0.00001
        self.gleak_max_value = 1000
        self.cm_t_min_value = 0.000001
        self.cm_t_max_value = 1000

        self.num_units = 16

        self.sensory_mu = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(0.3, 0.8))
        self.sensory_sigma = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(3.0, 8.0))
        self.sensory_W = nn.Parameter(torch.FloatTensor(state_dim, self.num_units).uniform_(self.w_init_min, self.w_init_max))

        sensory_erev_init = 2 * torch.randint(0, 2, (state_dim, self.num_units)) - 1
        self.sensory_erev = nn.Parameter(sensory_erev_init.float() * self.erev_init_factor)

        self.mu = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(0.3, 0.8))
        self.sigma = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(3.0, 8.0))
        self.W = nn.Parameter(torch.FloatTensor(self.num_units, self.num_units).uniform_(self.w_init_min, self.w_init_max))

        erev_init = 2 * torch.randint(0, 2, (self.num_units, self.num_units)) - 1
        self.erev = nn.Parameter(erev_init.float() * self.erev_init_factor)

        self.vleak = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(-0.2, 0.2))
        self.gleak = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(self.gleak_init_min, self.gleak_init_max))
        self.cm_t = nn.Parameter(torch.FloatTensor(self.num_units).uniform_(self.cm_init_min, self.cm_init_max))

        self.actor = nn.Linear(self.num_units, action_dim)
        self.critic = nn.Linear(self.num_units, 1)

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_step(self, inputs, state):
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self.ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(self.action_var.shape, new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))

        action_mean = self.actor(ltc_features)
        dist = Normal(action_mean, torch.sqrt(self.action_var))

        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        ltc_features = self._ode_step(state, torch.zeros(state.shape[0], self.num_units).to(self.device))

        action_mean = self.actor(ltc_features)

        action_var = self.action_var.expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))

        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)

        state_value = self.critic(ltc_features).squeeze()

        return action_logprobs, state_value, dist_entropy

    def clip_parameters(self):
        with torch.no_grad():
            self.W.clamp_(self.w_min_value, self.w_max_value)
            self.sensory_W.clamp_(self.w_min_value, self.w_max_value)
            self.gleak.clamp_(self.gleak_min_value, self.gleak_max_value)
            self.cm_t.clamp_(self.cm_t_min_value, self.cm_t_max_value)


class DeepActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(DeepActorCritic, self).__init__()
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(self.action_var.shape, new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
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
        pass


class WideActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(WideActorCritic, self).__init__()
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(self.action_var.shape, new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
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
        pass


class PPO:
    def __init__(self, state_dim, action_dim, action_std_init, lr, gamma, K_epochs, eps_clip, device, network_type="standard"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.network_type = network_type

        # 根据网络类型选择相应的网络架构
        if network_type == "ltc":
            network_class = LTCActorCritic
        elif network_type == "deep":
            network_class = DeepActorCritic
        elif network_type == "wide":
            network_class = WideActorCritic
        else:
            network_class = ActorCritic

        self.policy = network_class(state_dim, action_dim, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = network_class(state_dim, action_dim, action_std_init, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        if action.shape[0] == 1:
            return action.cpu().numpy().flatten(), action_logprob
        else:
            return action.cpu().numpy(), action_logprob


def load_protagonist_model(model_path, device):
    """加载训练好的主角模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    print(f"正在加载模型: {model_path}")
    model_info = torch.load(model_path, map_location=device)

    # 提取模型参数
    state_dim = model_info.get('state_dim')
    action_dim = model_info.get('action_dim')
    action_std = model_info.get('action_std', 0.6)
    network_type = model_info.get('network_type', 'standard')

    # 创建PPO对象
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
        device=device,
        network_type=network_type
    )

    # 加载权重
    if 'policy_state_dict' in model_info:
        protagonist.policy.load_state_dict(model_info['policy_state_dict'])
        protagonist.policy_old.load_state_dict(model_info['policy_state_dict'])
        print(f"成功加载{network_type}模型权重")
    else:
        print("警告: 模型文件中没有找到策略权重!")

    return protagonist, model_info


def extract_vehicle_info(env, info):
    """从环境中提取车辆详细信息"""
    vehicle_info = {}

    try:
        # 获取主车辆对象
        vehicle = env.vehicle

        # 速度信息 (km/h)
        vehicle_info['speed_kmh'] = vehicle.speed_km_h if hasattr(vehicle, 'speed_km_h') else 0

        # 加速度 (m/s^2)
        if hasattr(vehicle, 'last_speed') and hasattr(vehicle, 'speed'):
            vehicle_info['acceleration'] = (vehicle.speed - vehicle.last_speed) * 10  # 假设10Hz更新
        else:
            vehicle_info['acceleration'] = 0

        # 车头时距 (Time Headway, 秒)
        if hasattr(vehicle, 'lidar') and hasattr(vehicle.lidar, 'get_surrounding_vehicles'):
            surrounding_vehicles = vehicle.lidar.get_surrounding_vehicles()
            if surrounding_vehicles:
                # 找到前方最近的车辆
                min_distance = float('inf')
                for other_vehicle in surrounding_vehicles:
                    distance = vehicle.position.distance(other_vehicle.position)
                    if distance < min_distance:
                        min_distance = distance

                # 车头时距 = 距离 / 速度
                if vehicle.speed > 0.1:  # 避免除以零
                    vehicle_info['time_headway'] = min_distance / vehicle.speed
                else:
                    vehicle_info['time_headway'] = float('inf')
                vehicle_info['front_vehicle_distance'] = min_distance
            else:
                vehicle_info['time_headway'] = float('inf')
                vehicle_info['front_vehicle_distance'] = float('inf')
        else:
            vehicle_info['time_headway'] = float('inf')
            vehicle_info['front_vehicle_distance'] = float('inf')

        # 位置信息
        if hasattr(vehicle, 'position'):
            vehicle_info['position_x'] = vehicle.position[0]
            vehicle_info['position_y'] = vehicle.position[1]
        else:
            vehicle_info['position_x'] = 0
            vehicle_info['position_y'] = 0

        # 航向角 (度)
        if hasattr(vehicle, 'heading_theta'):
            vehicle_info['heading_deg'] = np.degrees(vehicle.heading_theta)
        else:
            vehicle_info['heading_deg'] = 0

        # 横向偏移 (距离车道中心线的距离)
        if hasattr(vehicle, 'lane') and hasattr(vehicle.lane, 'local_coordinates'):
            lateral_offset, _ = vehicle.lane.local_coordinates(vehicle.position)
            vehicle_info['lateral_offset'] = lateral_offset
        else:
            vehicle_info['lateral_offset'] = 0

        # 转向角
        if hasattr(vehicle, 'steering'):
            vehicle_info['steering_angle'] = vehicle.steering
        else:
            vehicle_info['steering_angle'] = 0

        # 油门和刹车
        if hasattr(vehicle, 'throttle_brake'):
            vehicle_info['throttle_brake'] = vehicle.throttle_brake
        else:
            vehicle_info['throttle_brake'] = 0

    except Exception as e:
        print(f"提取车辆信息时出错: {e}")
        # 返回默认值
        vehicle_info = {
            'speed_kmh': 0,
            'acceleration': 0,
            'time_headway': float('inf'),
            'front_vehicle_distance': float('inf'),
            'position_x': 0,
            'position_y': 0,
            'heading_deg': 0,
            'lateral_offset': 0,
            'steering_angle': 0,
            'throttle_brake': 0
        }

    return vehicle_info


def test_protagonist_detailed(model_path, num_episodes=1, save_to_excel=True, difficulty="easy"):
    """
    测试主角模型并输出详细的环境信息

    Args:
        model_path: 模型文件路径
        num_episodes: 测试回合数
        save_to_excel: 是否保存到Excel
        difficulty: 测试难度级别
    """
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    protagonist, model_info = load_protagonist_model(model_path, device)

    # 配置环境参数
    if difficulty == "easy":
        dynamics_params = {
            "max_engine_force": 2350,
            "max_brake_force": 400,
            "wheel_friction": 1.4,
            "max_steering": 45,
            "mass": 900
        }
    elif difficulty == "medium":
        dynamics_params = {
            "max_engine_force": 2350,
            "max_brake_force": 400,
            "wheel_friction": 1.4,
            "max_steering": 45,
            "mass": 900
        }
    else:  # hard or total
        dynamics_params = {
            "max_engine_force": 2350,
            "max_brake_force": 400,
            "wheel_friction": 1.4,
            "max_steering": 45,
            "mass": 900
        }

    # 创建环境
    env_config = {
        "num_scenarios": 1,
        "horizon": 1000,
        "random_dynamics": {},
        "show_logo": False,
        "show_interface": False,
        "debug": False,
        "log_level": 50
    }

    # 设置动力学参数
    for param_name, value in dynamics_params.items():
        env_config["random_dynamics"][param_name] = (value, value)

    env = VaryingDynamicsEnv(env_config)

    # 存储所有步骤的数据
    all_steps_data = []

    print("=" * 80)
    print(f"开始测试模型: {os.path.basename(model_path)}")
    print(f"测试回合数: {num_episodes}")
    print(f"难度级别: {difficulty}")
    print("=" * 80)

    for episode in range(num_episodes):
        print(f"\n===== 回合 {episode + 1} =====")

        # 重置环境
        state, _ = env.reset()

        episode_reward = 0
        episode_length = 0

        for step in range(env.config["horizon"]):
            # 主角选择动作
            action, _ = protagonist.select_action(state)

            # 环境交互
            next_state, reward, terminated, truncated, info = env.step(action)

            # 提取车辆详细信息
            vehicle_info = extract_vehicle_info(env, info)

            # 记录当前步骤的所有信息
            step_data = {
                'episode': episode + 1,
                'step': step + 1,
                'reward': reward,
                'cumulative_reward': episode_reward + reward,
                'action_0': action[0],
                'action_1': action[1] if len(action) > 1 else 0,
                **vehicle_info,  # 添加所有车辆信息
                'terminated': terminated,
                'truncated': truncated,
                'arrive_dest': info.get('arrive_dest', False),
                'crash': info.get('crash', False),
                'out_of_road': info.get('out_of_road', False)
            }

            all_steps_data.append(step_data)

            # 每10步输出一次信息
            if (step + 1) % 10 == 0:
                print(f"步骤 {step + 1:4d} | "
                      f"速度: {vehicle_info['speed_kmh']:6.2f} km/h | "
                      f"加速度: {vehicle_info['acceleration']:6.2f} m/s² | "
                      f"车头时距: {vehicle_info['time_headway']:6.2f} s | "
                      f"奖励: {reward:6.2f}")

            state = next_state
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        # 输出回合总结
        print(f"\n回合 {episode + 1} 完成:")
        print(f"  总步数: {episode_length}")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  到达终点: {info.get('arrive_dest', False)}")
        print(f"  发生碰撞: {info.get('crash', False)}")
        print(f"  驶出道路: {info.get('out_of_road', False)}")

    # 保存到Excel
    if save_to_excel and all_steps_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = f"logs/test_detailed_{timestamp}.xlsx"

        # 创建保存目录
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 转换为DataFrame并保存
        df = pd.DataFrame(all_steps_data)

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='DetailedSteps', index=False)

            # 添加统计摘要
            summary_data = {
                '指标': ['平均速度 (km/h)', '最大速度 (km/h)', '最小速度 (km/h)',
                        '平均加速度 (m/s²)', '最大加速度 (m/s²)', '最小加速度 (m/s²)',
                        '平均车头时距 (s)', '最小车头时距 (s)',
                        '总奖励', '总步数'],
                '数值': [
                    df['speed_kmh'].mean(),
                    df['speed_kmh'].max(),
                    df['speed_kmh'].min(),
                    df['acceleration'].mean(),
                    df['acceleration'].max(),
                    df['acceleration'].min(),
                    df[df['time_headway'] != float('inf')]['time_headway'].mean() if len(df[df['time_headway'] != float('inf')]) > 0 else 0,
                    df[df['time_headway'] != float('inf')]['time_headway'].min() if len(df[df['time_headway'] != float('inf')]) > 0 else 0,
                    df['cumulative_reward'].iloc[-1] if len(df) > 0 else 0,
                    len(df)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"\n详细测试数据已保存至: {excel_file}")

    # 关闭环境
    env.close()

    print("\n测试完成!")

    return all_steps_data


def parse_args():
    parser = argparse.ArgumentParser(description="测试训练好的主角模型并输出详细信息")
    parser.add_argument("--model", type=str, required=True,
                        help="主角模型文件路径 (.pt文件)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="测试回合数 (默认: 1)")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard", "total"],
                        help="测试难度级别 (默认: easy)")
    parser.add_argument("--no_save", action="store_true",
                        help="不保存到Excel文件")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_protagonist_detailed(
        model_path=args.model,
        num_episodes=args.episodes,
        save_to_excel=not args.no_save,
        difficulty=args.difficulty
    )
