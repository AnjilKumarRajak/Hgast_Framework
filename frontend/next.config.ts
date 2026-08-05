import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: [
      'localhost:3000', 
      'belle-assists-misc-johnson.trycloudflare.com'  
  ]
};

export default nextConfig;
