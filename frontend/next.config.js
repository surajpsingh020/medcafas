/** @type {import('next').NextConfig} */
const nextConfig = {
  // API routes in app/api/ proxy to FastAPI directly — no rewrites needed.
  // FASTAPI_URL env var can override the default localhost:8000 for production.
};

module.exports = nextConfig;
