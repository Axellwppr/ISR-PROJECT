import mujoco

model = mujoco.MjModel.from_xml_path(
    "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/urdf/rel2.urdf",
)

breakpoint()

model.mj_saveLastXML(
    "/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/robots/rel2/xbot.xml"
)
