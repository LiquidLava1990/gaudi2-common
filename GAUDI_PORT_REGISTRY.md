# Gaudi Port Registry

Last verified: 2026-07-08

## Rule

Every Gaudi-hosted model/container that exposes a LAN service gets its own
unique host port. The owner explicitly promoted the current champion container to
`8000`; do not reuse `8000` for any additional Gaudi model API.

Reserved Gaudi model/container API range: `18274-18320`.

Allocation rule:

1. Check the live server listeners.
2. Check this registry.
3. Assign the first free port in `18274-18320`.
4. Write the assigned port into the container launch script and Trident config.
5. Verify the new port is open and any retired old port is closed.

## Assigned Ports

| Port | Status | Container / service | Endpoint | Notes |
| --- | --- | --- | --- | --- |
| 8000 | active primary | vllm-glm52-champion-ablid-maxctx-offload | http://<SERVER_IP>:8000/v1 | GLM-5.2 champion ablid max `glm52`, current Gaudi primary. |
| 18273 | legacy/offline | vllm-glm52-coresident-offload | http://<SERVER_IP>:18273/v1 | Previous GLM-5.2 container, retained as rollback only. Not in Trident discovery. |
| 18274 | next-free | reserved for next Gaudi model container |  | Use this for the next new Gaudi container if still free. |

## Retired / Do Not Use

| Port | Reason |
| --- | --- |
| 8000 | Current champion primary only. Do not allocate to any new Gaudi container. |

## Non-Model Server Ports

| Port | Service | Notes |
| --- | --- | --- |
| 22 | SSH | Admin over LAN/Tailscale. |
| 9000 | Portainer | Existing admin UI. |
