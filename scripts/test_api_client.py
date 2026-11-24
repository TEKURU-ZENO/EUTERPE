import asyncio
import time
import httpx
from euterpe.api.main import app

async def main():
    print("[1] Initializing ASGITransport...")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        print("[2] Testing /api/health...")
        t0 = time.time()
        r = await client.get("/api/health")
        print(f"  Health: status={r.status_code}, latency={time.time() - t0:.4f}s, body={r.json()['status']}")

        print("[3] Testing /api/discover...")
        t0 = time.time()
        r = await client.get("/api/discover?user_id=user_000001&limit=5")
        data = r.json()
        print(f"  Discover: status={r.status_code}, latency={time.time() - t0:.4f}s, items={len(data['recommendations'])}, latency_ms={data['latency_ms']}")

        print("[4] Testing /api/hidden-gems...")
        t0 = time.time()
        r = await client.get("/api/hidden-gems?user_id=user_000001&limit=5")
        data = r.json()
        print(f"  Hidden Gems: status={r.status_code}, latency={time.time() - t0:.4f}s, items={len(data['gems'])}")

        print("[5] Testing /api/universe...")
        t0 = time.time()
        r = await client.get("/api/universe?limit=10&user_id=user_000001")
        data = r.json()
        print(f"  Universe: status={r.status_code}, latency={time.time() - t0:.4f}s, points={len(data['points'])}")

        print("[6] Testing /api/taste-profile...")
        t0 = time.time()
        r = await client.get("/api/taste-profile?user_id=user_000001")
        data = r.json()
        print(f"  Taste Profile: status={r.status_code}, latency={time.time() - t0:.4f}s, openness={data['openness_score']}, mainstream={data['mainstream_index']}")

        print("[7] Testing /api/lab...")
        t0 = time.time()
        r = await client.get("/api/lab?user_id=user_000001&alpha=0.6&beta=0.3&max_per_artist=2&limit=5")
        data = r.json()
        print(f"  Lab: status={r.status_code}, latency={time.time() - t0:.4f}s, deltas={data['live_deltas']}")

    print("\nALL API ENDPOINTS TESTED SUCCESSFULLY VIA ASYNC ASGI!")

if __name__ == "__main__":
    asyncio.run(main())
