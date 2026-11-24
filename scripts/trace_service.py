import sys
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("1. Importing service...")
from euterpe.api.service import get_discovery_service

log("2. Initializing service...")
s = get_discovery_service()
log("Service initialized successfully!")

log("3. Testing resolve_user_idx...")
uid = s.resolve_user_idx("user_000001")
log(f"User resolved: {uid}")

log("4. Testing compute_discovery_recommendations...")
t0 = time.time()
tracks, expls, mode = s.compute_discovery_recommendations(user_idx=uid, limit=5)
log(f"Recommendations computed in {time.time()-t0:.4f}s! Tracks: {tracks}")

log("5. Testing get_hidden_gems...")
t0 = time.time()
gems = s.get_hidden_gems(user_idx=uid, limit=5)
log(f"Hidden gems computed in {time.time()-t0:.4f}s! Gems: {len(gems)}")

log("6. Testing get_universe_viewport...")
t0 = time.time()
pts = s.get_universe_viewport(limit=10)
log(f"Universe viewport computed in {time.time()-t0:.4f}s! Points: {len(pts)}")

log("7. Testing get_taste_profile...")
t0 = time.time()
tp = s.get_taste_profile(user_idx=uid)
log(f"Taste profile computed in {time.time()-t0:.4f}s! Openness: {tp['openness_score']}")

log("ALL SERVICE METHODS VERIFIED INSTANTLY!")
