// © 2025 – Shah Rukh Khan Pickup-Lines & Date-Locations MCP Server (TS Edition) with LLM Utility

import { createMcpHandler } from "mcp-handler";
import { z }              from "zod";

const TOKEN             = process.env.AUTH_TOKEN!;
const MY_NUMBER         = process.env.MY_NUMBER!;
const OPENAI_API_KEY    = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const HF_TOKEN          = process.env.HF_TOKEN;

if (!TOKEN || !MY_NUMBER) {
  throw new Error("Missing AUTH_TOKEN or MY_NUMBER");
}

type Provider = {
  name: string;
  endpoint: string;
  headers: Record<string,string>;
  model: string;
  priority: number;
};

class LLMUtility {
  private providers: Provider[];
  private maxRetries = 3;
  private retryDelay = 1000;

  constructor() {
    this.providers = this.initProviders();
  }

  private initProviders(): Provider[] {
    const list: Provider[] = [];
    if (OPENAI_API_KEY) list.push({
      name: "openai",
      endpoint: "https://api.openai.com/v1/chat/completions",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      model: "gpt-3.5-turbo",
      priority: 1
    });
    if (ANTHROPIC_API_KEY) list.push({
      name: "anthropic",
      endpoint: "https://api.anthropic.com/v1/messages",
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
      },
      model: "claude-3-haiku-20240307",
      priority: 2
    });
    if (HF_TOKEN) list.push({
      name: "huggingface",
      endpoint: "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json"
      },
      model: "microsoft/DialoGPT-small",
      priority: 3
    });
    return list.sort((a, b) => a.priority - b.priority);
  }

  async generateText(prompt: string, context: any = {}): Promise<{ text:string; provider:string; model:string }> {
    for (const p of this.providers) {
      try {
        const out = await this.callProvider(p, prompt, context);
        if (out) return { text: out, provider: p.name, model: p.model };
      } catch {
        /* try next */
      }
    }
    return { text: "🤖 (fallback)", provider: "fallback", model: "local" };
  }

  private async callProvider(p: Provider, prompt: string, context: any): Promise<string> {
    const payload = this.makePayload(p, prompt, context);
    for (let i = 1; i <= this.maxRetries; i++) {
      const resp = await fetch(p.endpoint, {
        method: "POST",
        headers: p.headers,
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30000)
      });
      if (resp.ok) {
        const data = await resp.json();
        return this.extract(p, data);
      }
      if (resp.status === 429) {
        await new Promise(r => setTimeout(r, this.retryDelay * i));
        continue;
      }
      throw new Error(`${p.name} HTTP ${resp.status}`);
    }
    throw new Error("Retries exhausted");
  }

  private makePayload(p: Provider, prompt: string, context: any) {
    switch (p.name) {
      case "openai":
        return {
          model: p.model,
          messages: [
            { role: "system", content: context.systemPrompt || "You are helpful." },
            { role: "user", content: prompt }
          ],
          max_tokens: context.maxTokens || 150,
          temperature: context.temperature || 0.8,
          top_p: 0.9
        };
      case "anthropic":
        return {
          model: p.model,
          system: context.systemPrompt || "You are helpful.",
          messages: [{ role: "user", content: prompt }],
          max_tokens: context.maxTokens || 150,
          temperature: context.temperature || 0.8
        };
      case "huggingface":
        return {
          inputs: prompt,
          parameters: {
            max_new_tokens: context.maxTokens || 150,
            temperature: context.temperature || 0.8,
            top_p: 0.9,
            do_sample: true
          }
        };
    }
  }

  private extract(p: Provider, data: any): string {
    if (p.name === "openai") return data.choices?.[0]?.message?.content || "";
    if (p.name === "anthropic") return data?.content?.?.text || "";
    if (p.name === "huggingface" && Array.isArray(data) && data?.generated_text) {
      return data.generated_text.replace(data.inputs || "", "").trim();
    }
    return "";
  }
}

const llm = new LLMUtility();
const USER_AGENT = "Puch/1.0";
const DDG_HTML   = "https://html.duckduckgo.com/html/?q=";

async function stripHtml(html: string): Promise<string> {
  return html.replace(/<[^>]+>/g, "").replace(/\s{2,}/g, " ").trim();
}

const handlerFactory = (server: any) => {
  server.tool("validate", "Return owner number", {}, async () => ({
    content: [{ type: "text", text: MY_NUMBER }]
  }));

  server.tool(
    "generate_srk_pickup_line",
    "SRK-style pickup line",
    { user_info: z.string(), target_info: z.string().optional() },
    async ({ user_info, target_info }) => {
      const prompt = `Write a romantic SRK pickup line for ${user_info}` +
                     (target_info ? ` about ${target_info}` : "");
      const res = await llm.generateText(prompt, {
        systemPrompt: "You are SRK, King of Romance",
        type: "pickup_line"
      });
      return { content: [{ type: "text", text: `💕 "${res.text}"\n\n*(${res.provider})*` }] };
    }
  );

  server.tool(
    "find_date_locations",
    "Suggest date spots",
    { city: z.string(), date_type: z.string().default("romantic"), budget: z.string().default("moderate") },
    async ({ city, date_type, budget }) => {
      const q = `best ${date_type} date spots ${city} ${budget}`;
      const r = await fetch(`${DDG_HTML}${encodeURIComponent(q)}`, { headers: { "User-Agent": USER_AGENT } });
      if (!r.ok) throw new Error("Search failed");
      const html = await r.text();
      const items = [...html.matchAll(/result__a href="([^"]+)">([^<]+)<\/a>[\s\S]*?result__snippet">([^<]+)/gi)]
        .slice(0,5)
        .map(([,url,title,snip])=>({ url, title: title.trim(), snippet: snip.trim().slice(0,200) }));
      const text = items.map((it,i)=>`**${i+1}. ${it.title}**\n${it.snippet}...\n🔗 ${it.url}`).join("\n\n");
      return { content: [{ type: "text", text: text || "No spots found" }] };
    }
  );

  server.tool(
    "generate_srk_flirty_reply",
    "SRK-style flirty reply",
    { message: z.string(), your_name: z.string().optional() },
    async ({ message, your_name }) => {
      const prompt = `Reply flirtatiously as SRK to: "${message}"` + (your_name ? ` from ${your_name}` : "");
      const res = await llm.generateText(prompt, {
        systemPrompt: "You are SRK, responding charmingly",
        type: "flirty_reply"
      });
      return { content: [{ type: "text", text: `💕 "${res.text}"\n\n*(${res.provider})*` }] };
    }
  );
};

export const { GET, POST } = createMcpHandler(
  handlerFactory,
  { basePath: "/api", auth: { type: "bearer", token: TOKEN } },
  { verboseLogs: true }
);

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Authorization, Content-Type"
    }
  });
}
