# emotion2vec API

CPU-only wrapper for the locally cached `iic/emotion2vec_plus_base` model.
It exposes the interface consumed by `content_analysis_api`:

- `GET /health`
- `POST /predict` with multipart field `file`

The returned emotion is a whole-audio observation. It must not be assigned to
an individual speaker unless a future per-speaker audio segmentation stage is
added.

