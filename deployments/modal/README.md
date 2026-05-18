# Deprecated — moved to `src/qr_sampler/connectors/modal/`

The Modal app, Dockerfile, and secrets reference moved into the installable
package so downstream consumers (e.g. `qr-llm-chat`) can depend on them via
`pip install qr-sampler[modal]`. Use:

```bash
modal deploy -m qr_sampler.connectors.modal.app
```

The thin `app.py` shim in this directory re-exports from the new location so
legacy `modal deploy deployments/modal/app.py` invocations keep working.
Scheduled for deletion in qr-sampler 0.5.0.
