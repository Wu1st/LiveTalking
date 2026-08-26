from __future__ import annotations

import unittest

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from server.document_proxy import setup_document_proxy_routes


class DocumentProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.received: dict[str, object] = {}
        upstream = web.Application()

        async def health(_: web.Request) -> web.Response:
            return web.json_response({"status": "ok", "service": "ocr-document"})

        async def create_job(request: web.Request) -> web.Response:
            form = await request.post()
            uploaded = form["file"]
            self.received = {
                "translate": request.query.get("translate"),
                "source_language": request.query.get("source_language"),
                "filename": uploaded.filename,
                "body": uploaded.file.read(),
            }
            return web.json_response({"job_id": "job-42", "status": "queued"}, status=202)

        async def create_translation(request: web.Request) -> web.Response:
            self.received = {
                "idempotency_key": request.headers.get("Idempotency-Key"),
                "json": await request.json(),
            }
            return web.json_response({"job_id": "translation-42"}, status=202)

        async def download(_: web.Request) -> web.Response:
            return web.Response(
                body=b"translated document",
                content_type="text/plain",
                headers={"Content-Disposition": 'attachment; filename="result.txt"'},
            )

        upstream.router.add_get("/health", health)
        upstream.router.add_post("/v1/document/jobs", create_job)
        upstream.router.add_post("/v1/translation/document-jobs", create_translation)
        upstream.router.add_get("/v1/document/jobs/{job_id}/download", download)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()

        proxy = web.Application(client_max_size=1024**2)
        setup_document_proxy_routes(proxy, base_url=str(self.upstream.make_url("/")).rstrip("/"))
        self.client = TestClient(TestServer(proxy))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.upstream.close()

    async def test_health_is_available_on_integrated_prefix(self) -> None:
        response = await self.client.get("/api/document/health")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["service"], "ocr-document")

    async def test_upload_streams_query_and_multipart_body(self) -> None:
        form = FormData()
        form.add_field(
            "file",
            b"%PDF-test",
            filename="sample.pdf",
            content_type="application/pdf",
        )
        response = await self.client.post(
            "/api/document/jobs?translate=true&source_language=en",
            data=form,
        )
        self.assertEqual(response.status, 202)
        self.assertEqual((await response.json())["job_id"], "job-42")
        self.assertEqual(
            self.received,
            {
                "translate": "true",
                "source_language": "en",
                "filename": "sample.pdf",
                "body": b"%PDF-test",
            },
        )

    async def test_translation_json_and_idempotency_key_are_forwarded(self) -> None:
        response = await self.client.post(
            "/v1/translation/document-jobs",
            json={"ocr_job_id": "a" * 32, "target_language": "zh"},
            headers={"Idempotency-Key": "request-7"},
        )
        self.assertEqual(response.status, 202)
        self.assertEqual(self.received["idempotency_key"], "request-7")
        self.assertEqual(self.received["json"]["target_language"], "zh")

    async def test_download_body_and_filename_are_preserved(self) -> None:
        response = await self.client.get(
            "/api/document/jobs/job-42/download?format=translated_txt"
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"translated document")
        self.assertEqual(
            response.headers["Content-Disposition"],
            'attachment; filename="result.txt"',
        )

    async def test_unavailable_upstream_returns_503_envelope(self) -> None:
        unavailable = web.Application()
        setup_document_proxy_routes(unavailable, base_url="http://127.0.0.1:1")
        client = TestClient(TestServer(unavailable))
        await client.start_server()
        try:
            response = await client.get("/api/document/health")
            payload = await response.json()
            self.assertEqual(response.status, 503)
            self.assertEqual(payload["code"], -1)
            self.assertEqual(payload["msg"], "OCR/translation service unavailable")
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
