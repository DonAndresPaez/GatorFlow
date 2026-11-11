'''aero: the aerodynamics that colour the model, run from inside TouchDesigner.

    Kinect -> KinectReader -> OSC pose -> geo1 (TouchDesigner)
                                            |
                         catalog.py   which model is on the plate (setup/model.json for now)
                         flow.py      the hardcoded wind, turned into the plate's frame by the pose
                         panel.py     pipeline 1: 3D panel method (traditional, exact for potential flow)
                         lifting.py   the same with lift for wings (doublets, Kutta, wake): opt-in per model
                         surrogate.py pipeline 2: pretrained network, per-frame prediction
                         flowlines.py streamlines over the model and the plate (from the panel method)
                         pipeline.py  pose in, mesh out: a colour per point, and flow-line ribbons

Only NumPy is needed at run time, because TouchDesigner's Python ships with it
and nothing else this needs. Training the network (train.py) runs in a normal
Python with PyTorch, outside TouchDesigner.

Not imported from common/: common.paths imports OpenCV, which TouchDesigner's
Python does not have.
'''
