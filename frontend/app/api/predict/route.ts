import { NextRequest, NextResponse } from 'next/server';

// Disable Next.js body size limit and response timeout for this route
export const dynamic = 'force-dynamic';
export const maxDuration = 300; // 5 minutes — enough for CPU inference

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const upstream = await fetch(`${FASTAPI_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // Node fetch has no default timeout — request will wait as long as needed
      signal: AbortSignal.timeout(300_000), // 5 min hard cap
    });

    const data = await upstream.json();

    if (!upstream.ok) {
      return NextResponse.json(data, { status: upstream.status });
    }

    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Proxy error';
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}
