import torch

CHASSIS_DIM = 9
TURRET_DIM = 3
FIRE_DIM = 2
FIRE_ACTION_ID = 27


def decompose_action(action: int, chassis_dim: int = CHASSIS_DIM, fire_action_id: int = FIRE_ACTION_ID):
    """
    将环境离散动作(0~27)分解为三头动作：
    - chassis: 0~8
    - turret: 0~2
    - fire: 0(不射击) / 1(射击)
    """
    if action == fire_action_id:
        return 0, 0, 1
    chassis = action % chassis_dim
    turret = action // chassis_dim
    return chassis, turret, 0


def compose_action(chassis: int, turret: int, fire: int, chassis_dim: int = CHASSIS_DIM, fire_action_id: int = FIRE_ACTION_ID):
    """
    将三头动作组装回环境动作空间(0~27)。
    fire=1 时统一映射到停步开火动作 27。
    """
    if fire == 1:
        return fire_action_id
    return int(turret) * int(chassis_dim) + int(chassis)


def decompose_action_tensor(actions: torch.Tensor, chassis_dim: int = CHASSIS_DIM, fire_action_id: int = FIRE_ACTION_ID):
    """
    actions: [B, T, 1] 或 [B, 1]
    返回: chassis_idx, turret_idx, fire_idx (shape 与 actions 相同)
    """
    action_ids = actions.long()

    fire_mask = (action_ids == fire_action_id)
    chassis_idx = torch.remainder(action_ids, chassis_dim)
    turret_idx = torch.div(action_ids, chassis_dim, rounding_mode='floor')
    fire_idx = torch.zeros_like(action_ids)

    chassis_idx = torch.where(fire_mask, torch.zeros_like(chassis_idx), chassis_idx)
    turret_idx = torch.where(fire_mask, torch.zeros_like(turret_idx), turret_idx)
    fire_idx = torch.where(fire_mask, torch.ones_like(fire_idx), fire_idx)

    return chassis_idx, turret_idx, fire_idx
