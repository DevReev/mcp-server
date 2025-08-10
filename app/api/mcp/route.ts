// © 2025 – Shah Rukh Khan Pickup-Lines & Date-Locations MCP Server (JS Edition) with LLM Utility

/* ──────────────────────────────────────────────────────────────
   0.  Runtime & Deps
────────────────────────────────────────────────────────────── */
import { createMcpHandler } from "mcp-handler";
import { z } from "zod";

/* ──────────────────────────────────────────────────────────────
   1.  Environment
────────────────────────────────────────────────────────────── */
const TOKEN = process.env.AUTH_TOKEN;
const MY_NUMBER = process.env.MY_NUMBER;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const HF_TOKEN = process.env.HF_TOKEN;

if (!TOKEN || !MY_NUMBER) {
  throw new Error("Missing AUTH_TOKEN or MY_NUMBER env variable");
}

/* ──────────────────────────────────────────────────────────────
   2.  LLM Utility Class
────────────────────────────────────────────────────────────── */
class LLMUtility {
  constructor() {
    this.providers = this.initProviders();
    this.fallbackEnabled = true;
    this.maxRetries = 3;
    this.retryDelay = 1000;
  }

  initProviders() {
    const list = [];
    if (OPENAI_API_KEY)
      list.push({
        name: "openai",
        endpoint: "https://api.openai.com/v1/chat/completions",
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        model: "gpt-3.5-turbo",
        priority: 1,
      });
    if (ANTHROPIC_API_KEY)
      list.push({
        name: "anthropic",
        endpoint: "https://api.anthropic.com/v1/messages",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "Content-Type": "application/json",
          "anthropic-version": "2023-06-01",
        },
        model: "claude-3-haiku-20240307",
        priority: 2,
      });
    if (HF_TOKEN)
      list.push({
        name: "huggingface",
        endpoint:
          "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        model: "microsoft/DialoGPT-small",
        priority: 3,
      });
    return list.sort((a, b) => a.priority - b.priority);
  }

  async generateText(prompt, context = {}) {
    if (!this.providers.length) {
      return {
        text: this.fallback(prompt, context),
        provider: "fallback",
        model: "local",
      };
    }
    for (const p of this.providers) {
      try {
        const out = await this.call(p, prompt, context);
        if (out) return { text: out, provider: p.name, model: p.model };
      } catch (err) {
        console.warn(`LLM ${p.name} failed:`, err.message);
      }
    }
    return {
      text: this.fallback(prompt, context),
      provider: "fallback",
      model: "local",
    };
  }

  async call(provider, prompt, context) {
    const payload = this.makePayload(provider, prompt, context);
    for (let i = 1; i <= this.maxRetries; i++) {
      const resp = await fetch(provider.endpoint, {
        method: "POST",
        headers: provider.headers,
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000),
      });
      if (resp.ok) {
        const data = await resp.json();
        return this.extract(provider, data);
      }
      if (resp.status === 429) {
        await this.delay(this.retryDelay * i);
        continue;
      }
      throw new Error(`${provider.name} HTTP ${resp.status}`);
    }
  }

  makePayload(provider, prompt, context) {
    switch (provider.name) {
      case "openai":
        return {
          model: provider.model,
          messages: [
            {
              role: "system",
              content: context.systemPrompt || "You are a helpful assistant.",
            },
            { role: "user", content: prompt },
          ],
          max_tokens: context.maxTokens || 150,
          temperature: context.temperature || 0.8,
          top_p: 0.9,
        };
      case "anthropic":
        return {
          model: provider.model,
          system: context.systemPrompt || "You are a helpful assistant.",
          messages: [{ role: "user", content: prompt }],
          max_tokens: context.maxTokens || 150,
          temperature: context.temperature || 0.8,
        };
      case "huggingface":
        return {
          inputs: prompt,
          parameters: {
            max_new_tokens: context.maxTokens || 150,
            temperature: context.temperature || 0.8,
            top_p: 0.9,
            do_sample: true,
          },
        };
    }
  }

  extract(provider, data) {
    if (provider.name === "openai")
      return data.choices?.[0]?.message?.content || "";
    if (provider.name === "anthropic") return data?.content?.[0]?.text || "";
    if (
      provider.name === "huggingface" &&
      Array.isArray(data) &&
      data[0]?.generated_text
    ) {
      return data[0].generated_text.replace(data[0].inputs || "", "").trim();
    }
    return "";
  }

  fallback(prompt, context) {
    return "🤖 (fallback response)";
  }

  delay(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }
}

/* ──────────────────────────────────────────────────────────────
   3.  Utility Helpers
────────────────────────────────────────────────────────────── */
const USER_AGENT = "Puch/1.0 (Autonomous)";
const DDG_HTML = "https://html.duckduckgo.com/html/?q=";
const llm = new LLMUtility();

async function stripHtml(html) {
  return html
    .replace(/<[^>]+>/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

/* ──────────────────────────────────────────────────────────────
   4.  MCP Handler
────────────────────────────────────────────────────────────── */
const handlerFactory = (server) => {
  // 4.1 Validate
  server.tool("validate", "Return owner number", {}, async () => ({
    content: [{ type: "text", text: MY_NUMBER }],
  }));

  // 4.2 SRK Pickup Line
  server.tool(
    "generate_srk_pickup_line",
    "SRK-style pickup line",
    {
      user_info: z.string(),
      target_info: z.string().optional(),
    },
    async ({ user_info, target_info }) => {
      const prompt =
        `Write a romantic SRK pickup line for ${user_info}` +
        (target_info ? ` about ${target_info}` : "");
      const res = await llm.generateText(prompt, {
        systemPrompt: "You are SRK, King of Romance",
        type: "pickup_line",
      });
      return {
        content: [
          {
            type: "text",
            text: `💕 "${res.text}"\n\n*(${res.provider})*`,
          },
        ],
      };
    }
  );

  // 4.3 Date Locations
  server.tool(
    "find_date_locations",
    "Suggest date spots",
    {
      city: z.string(),
      date_type: z.string().default("romantic"),
      budget: z.string().default("moderate"),
    },
    async ({ city, date_type, budget }) => {
      const q = `best ${date_type} date spots ${city} ${budget}`;
      const r = await fetch(`${DDG_HTML}${encodeURIComponent(q)}`, {
        headers: { "User-Agent": USER_AGENT },
      });
      if (!r.ok) throw new Error("Search failed");
      const html = await r.text();
      const items = [
        ...html.matchAll(
          /result__a href="([^"]+)">([^<]+)<\/a>[\s\S]*?result__snippet">([^<]+)/gi
        ),
      ]
        .slice(0, 5)
        .map(([, url, title, snip]) => ({
          url,
          title: title.trim(),
          snippet: snip.trim().slice(0, 200),
        }));
      const text = items
        .map(
          (it, i) => `**${i + 1}. ${it.title}**\n${it.snippet}...\n🔗 ${it.url}`
        )
        .join("\n\n");
      return { content: [{ type: "text", text: text || "No spots found" }] };
    }
  );

  // 4.4 Flirty Reply
  server.tool(
    "generate_srk_flirty_reply",
    "SRK-style flirty reply",
    {
      message: z.string(),
      your_name: z.string().optional(),
    },
    async ({ message, your_name }) => {
      const prompt =
        `Reply flirtatiously as SRK to: "${message}"` +
        (your_name ? ` from ${your_name}` : "");
      const res = await llm.generateText(prompt, {
        systemPrompt: "You are SRK, responding charmingly",
        type: "flirty_reply",
      });
      return {
        content: [
          {
            type: "text",
            text: `💕 "${res.text}"\n\n*(${res.provider})*`,
          },
        ],
      };
    }
  );
};

/* ──────────────────────────────────────────────────────────────
   5.  Exports for Vercel/Cloudflare
────────────────────────────────────────────────────────────── */
export async function GET() {
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

export const { GET, POST } = createMcpHandler(
  handlerFactory,
  { basePath: "/api", auth: { type: "bearer", token: TOKEN } },
  { verboseLogs: true }
);

// CORS preflight support
export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Authorization, Content-Type",
    },
  });
}
