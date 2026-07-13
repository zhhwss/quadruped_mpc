"""从 a2_scene.xml 提取所有物理参数，验证实际质量/惯量/连杆长度"""
import xml.etree.ElementTree as ET
import numpy as np

xml_path = '/Users/blackzhou/work/robot/mpc/test/quadruped_mpc/models/a2_scene.xml'
tree = ET.parse(xml_path)
root = tree.getroot()

print("=" * 60)
print("A2 物理参数解析报告 (来自 a2_scene.xml)")
print("=" * 60)

# ── 全局惯性系 ───────────────────────────────────────
option = root.find('option')
gravity = option.get('gravity') if option is not None else '0 0 -9.81'
print(f"\n[全局] gravity = {gravity}")

# ── base_link ───────────────────────────────────────
base = root.find(".//body[@name='base_link']")
inertial = base.find('inertial')
base_mass = float(inertial.get('mass'))
base_pos = np.array([float(x) for x in inertial.get('pos').split()])
base_quat = np.array([float(x) for x in inertial.get('quat').split()])
diag = np.array([float(x) for x in inertial.get('diaginertia').split()])
print(f"\n[base_link]")
print(f"  mass    = {base_mass:.3f} kg")
print(f"  pos     = {base_pos}")
print(f"  quat    = {base_quat}")
print(f"  diaginertia = {diag}")

# ── 所有 body 遍历 ───────────────────────────────────
links = {}  # name → dict
worldbody = root.find('worldbody')

for body in worldbody.findall('.//body'):
    name = body.get('name')
    inertial = body.find('inertial')
    if inertial is None:
        continue
    mass = float(inertial.get('mass'))
    pos = np.array([float(x) for x in inertial.get('pos').split()])
    quat = np.array([float(x) for x in inertial.get('quat').split()])
    diag = np.array([float(x) for x in inertial.get('diaginertia').split()])
    parent = body.get('name', '')  # 简化：直接用 body 名

    links[name] = {
        'mass': mass,
        'pos': pos,
        'quat': quat,
        'diag': diag,
        'parent_pos': None  # 需要递归查找
    }

print(f"\n[所有连杆] 共 {len(links)} 个带惯量的 body:")
total_mass_no_base = 0
for name, info in links.items():
    if name == 'base_link':
        continue
    print(f"  {name:30s}: mass={info['mass']:.3f} kg, "
          f"local_pos={info['pos']}, diag={info['diag']}")
    total_mass_no_base += info['mass']

print(f"\n  腿连杆总质量 = {total_mass_no_base:.3f} kg")
print(f"  全身总质量 = {base_mass + total_mass_no_base:.3f} kg")

# ── 连杆树结构 ───────────────────────────────────────
print(f"\n[连杆树结构]")
for body in worldbody.findall('.//body'):
    name = body.get('name')
    pos = body.get('pos', '0 0 0')
    parent = body.get('name', 'worldbody')  # MuJoCo body 嵌套
    # 找父节点
    parent_elem = None
    for p in worldbody.findall('.//body'):
        for child in p.findall('body'):
            if child.get('name') == name:
                parent_elem = p.get('name')
                break
    if parent_elem:
        print(f"  {name:30s} → parent: {parent_elem:30s} pos={pos}")

# ── joint 轴 & 限位 ──────────────────────────────────
print(f"\n[Joint 参数]")
for joint in root.findall('.//joint'):
    name = joint.get('name')
    axis = joint.get('axis', 'N/A')
    rng = joint.get('range', 'N/A')
    frc_rng = joint.get('actuatorfrcrange', 'N/A')
    print(f"  {name:20s}: axis={axis}, range={rng}, actuatorfrcrange={frc_rng}")
