import asyncio
from api.routes import recompute_tulca
from models import RecomputeTulcaRequest

async def test():
    req = RecomputeTulcaRequest(
        class_weights=[{"w_tg": 1.0, "w_bw": 1.0, "w_bg": 1.0}],
        s_dim=4,
        v_dim=150,
        tulca_channel=0
    )
    try:
        res = await recompute_tulca(req)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
