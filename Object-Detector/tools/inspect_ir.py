# tools/inspect_ir.py
import os, sys, numpy as np
import openvino as ov

def find_models_root(start_dir):
    cur = os.path.abspath(start_dir)
    for _ in range(6):
        xml = os.path.join(cur, "models", "model_openvino.xml")
        if os.path.isfile(xml):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None

def npinfo(tag, arrlike):
    a = np.array(arrlike)
    flat = a.ravel()
    print(f"{tag}: shape={a.shape}, dtype={a.dtype}, "
          f"min={(float(flat.min()) if flat.size else 'n/a')}, "
          f"max={(float(flat.max()) if flat.size else 'n/a')}")
    return a

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = find_models_root(here)
    if not root:
        print("ERROR: Could not locate models/model_openvino.xml upward from", here)
        sys.exit(1)
    os.chdir(root)
    xml_path = os.path.join("models", "model_openvino.xml")
    print(f"[info] Using root: {root}")
    print(f"[info] XML: {xml_path}")

    core = ov.Core()
    model = core.read_model(xml_path)

    print("== Inputs ==")
    for i, inp in enumerate(model.inputs):
        try:
            name = inp.any_name
        except Exception:
            name = f"inp[{i}]"
        print(f"  {i}: name={name}, shape={inp.partial_shape}, type={inp.get_element_type()}")

    print("== Outputs ==")
    for i, out in enumerate(model.outputs):
        # Some IRs have no names on outputs; don't call any_name to avoid crash
        print(f"  {i}: shape={out.partial_shape}, type={out.get_element_type()}")

    # Dummy (1,3,640,640) NCHW float32/255
    H, W = 640, 640
    dummy = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_nchw = np.transpose(dummy.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    bgr_nchw = np.transpose(dummy[..., ::-1].astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

    compiled = core.compile_model(model, "CPU")
    infer = compiled.create_infer_request()

    print("\n== Forward RGB/255 NCHW ==")
    res = infer.infer({compiled.input(0): rgb_nchw})
    outs_rgb = []
    for i, out in enumerate(compiled.outputs):
        a = npinfo(f"  out[{i}]", res[out])
        outs_rgb.append(a)

    print("\n== Forward BGR/255 NCHW ==")
    res2 = infer.infer({compiled.input(0): bgr_nchw})
    for i, out in enumerate(compiled.outputs):
        npinfo(f"  outB[{i}]", res2[out])

    if outs_rgb:
        largest = max(outs_rgb, key=lambda z: z.size)
        v = largest.reshape(-1)
        print("\nSample values (first 10) from largest RGB tensor:", v[:10])

if __name__ == "__main__":
    main()
