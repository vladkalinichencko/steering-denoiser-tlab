"""transformer_lens ругается, что на MPS результаты могут быть молча неверными.

Проверяем это, а не верим на слово: те же промпты, тот же стиринг, два устройства.
Если логиты и лоссы совпадают — предупреждение к нашей нагрузке не относится.
"""
import os, sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
import steering

texts = ["The weather today is quite pleasant and I", "He opened the door and saw"]
out = {}
for dev in ("cpu", "mps"):
    m = steering.load_model(dev)
    v = steering.vector("diffmean:sentiment", m, dev)
    row = []
    for alpha in (0.0, 20.0, 80.0, 160.0):
        hooks = [(steering.HOOK, steering.make_hook(v, alpha))]
        with torch.no_grad(), m.hooks(fwd_hooks=hooks):
            losses = [float(m(m.to_tokens(t), return_type="loss")) for t in texts]
        row.append((alpha, [round(x, 5) for x in losses]))
    out[dev] = row
    del m
for (a, c), (_, g) in zip(out["cpu"], out["mps"]):
    d = max(abs(x - y) for x, y in zip(c, g))
    print(f"alpha={a:6.1f}  cpu {c}  mps {g}  максимальное расхождение {d:.2e}")
os._exit(0)
