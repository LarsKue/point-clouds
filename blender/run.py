
# TODO: get script from args
script = "blender/render.py"
open_in_background = True

args = ["blender", "-P", script]

if open_in_background:
    args.append("-b")

if __name__ == "__main__":
    import os
    os.system(" ".join(args))
