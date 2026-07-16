import mujoco

model = mujoco.MjModel.from_xml_path('models/a2_scene.xml')

print("Joint Names and Indices:")
for i in range(model.njnt):
    name = model.joint(i).name
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i]
    print(f"Joint {i}: name='{name}' | qpos_adr={qpos_adr} | dof_adr={dof_adr}")
