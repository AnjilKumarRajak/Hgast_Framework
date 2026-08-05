import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ?? "http://10.203.3.145:7861";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();

    const backendRes = await fetch(`${BACKEND_URL}/translate`, {
      method: "POST",
      body: formData,
    });

    if (!backendRes.ok) {
      console.error("Backend error:", await backendRes.text());
      return NextResponse.json(
        { error: "Backend server error" },
        { status: backendRes.status }
      );
    }

    const data = await backendRes.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("Proxy error:", error);
    return NextResponse.json(
      { error: "Failed to process translation request" },
      { status: 500 }
    );
  }
}
