'''
    此处用于定义一些全局可用的函数
'''
import pygame
import os
import re

_GAME_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GAME_ROOT)


def load_image(image_path):
    candidates = [image_path]
    if image_path and not os.path.isabs(image_path):
        candidates.append(os.path.join(_PROJECT_ROOT, image_path))
        if image_path.startswith(("Bullet/", "Map/", "Unit/")):
            candidates.append(os.path.join(_PROJECT_ROOT, "game", image_path))

    for candidate in candidates:
        try:
            if candidate and os.path.exists(candidate):
                return pygame.image.load(candidate)
        except Exception:
            continue

    print(f"Warning: Cannot load image: {image_path}")
    return None
    
def get_next_filename(folder_path, prefix, extension='.txt'):
    """
    获取文件夹中最小的未被使用的文件名
    示例： next_filename = get_next_filename('game/Map/saved', 'map', '.txt')
          如果 game/Map/saved 文件夹中已有 map0.txt, map1.txt, map2.txt，则 next_filename 为 map3.txt
          结果不包含路径
    """
    # 确保文件夹路径存在
    os.makedirs(folder_path, exist_ok=True)
    
    # 获取所有文件
    files = [f for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f))]
    
    # 提取已存在的编号
    existing_numbers = []
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+){re.escape(extension)}$')
    
    for file in files:
        match = pattern.match(file)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # 找到最小可用的编号
    if not existing_numbers:
        next_num = 0
    else:
        existing_numbers.sort()
        # 寻找缺失的编号
        for i in range(len(existing_numbers)):
            if i != existing_numbers[i]:
                next_num = i
                break
        else:
            next_num = existing_numbers[-1] + 1
    
    # 返回完整路径
    return f"{prefix}{next_num}{extension}"

def get_class_from_str(class_name: str):
    
    import game.Bullet.NormalShell.NormalShell
    import game.Bullet.RocketShell.RocketShell
    import game.Bullet.HeavyShell.HeavyShell
    
    if class_name == 'normal_shell':
        return game.Bullet.NormalShell.NormalShell.NormalShell
    elif class_name == 'rocket_shell':
        return game.Bullet.RocketShell.RocketShell.RocketShell
    elif class_name == 'heavy_shell':
        return game.Bullet.HeavyShell.HeavyShell.HeavyShell
    else:
        return None
    
def count_distance(a, b):
    return ((a.position[0] - b.position[0]) ** 2 + (a.position[1] - b.position[1]) ** 2) ** 0.5

def set_font():
    # 字体设置
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
    ]
    font = None
    for path in font_paths:
        try:
            font = pygame.font.Font(path, 24)
            break
        except:
            continue
    if font is None:
        font = pygame.font.Font(None, 24)  # 使用默认字体
    return font

class Action:
    """单位的基本动作"""
    def __init__(self, forward = False, backward = False, left = False, right = False, mouse_pos = (0,0), fire = False, switch_ammo = False):
        self.forward = forward
        self.backward = backward
        self.left = left
        self.right = right
        self.mouse_pos = mouse_pos
        self.fire = fire
        self.switch_ammo = switch_ammo
    

class MethodGroup:
    """方法分组器"""
    
    def __init__(self, group_name):
        self.group_name = group_name
        self.methods = {}
    
    def __call__(self, func):
        self.methods[func.__name__] = func
        func._group = self.group_name
        return func
    
    def get_method_names(self):
        return list(self.methods.keys())
    
    def execute_all(self, instance):
        results = {}
        for name, method in self.methods.items():
            bound_method = getattr(instance, name, None)
            if bound_method:
                results[name] = bound_method()
        return results
