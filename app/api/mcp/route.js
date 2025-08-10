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

// LLM API Keys (optional - for enhanced generation)
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const HF_TOKEN = process.env.HF_TOKEN;

if (!TOKEN || !MY_NUMBER) {
  throw new Error("Missing AUTH_TOKEN or MY_NUMBER env variable");
}

/* ──────────────────────────────────────────────────────────────
   2.  LLM Utility Class with Fallback Strategy
   ────────────────────────────────────────────────────────────── */
class LLMUtility {
  constructor() {
    this.providers = this.initializeProviders();
    this.fallbackEnabled = true;
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1 second
  }

  initializeProviders() {
    const providers = [];

    // OpenAI Provider (Primary)
    if (OPENAI_API_KEY) {
      providers.push({
        name: "openai",
        endpoint: "https://api.openai.com/v1/chat/completions",
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        model: "gpt-3.5-turbo",
        priority: 1,
      });
    }

    // Anthropic Provider (Secondary)
    if (ANTHROPIC_API_KEY) {
      providers.push({
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
    }

    // Hugging Face Serverless (Tertiary)
    if (HF_TOKEN) {
      providers.push({
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
    }

    return providers.sort((a, b) => a.priority - b.priority);
  }

  async generateText(prompt, context = {}) {
    if (this.providers.length === 0) {
      return this.getFallbackResponse(prompt, context);
    }

    for (const provider of this.providers) {
      try {
        const response = await this.callProvider(provider, prompt, context);
        if (response && response.trim().length > 0) {
          return {
            text: response,
            provider: provider.name,
            model: provider.model,
          };
        }
      } catch (error) {
        console.warn(`Provider ${provider.name} failed:`, error.message);
        if (!this.fallbackEnabled) {
          throw error;
        }
        // Continue to next provider
      }
    }

    // All providers failed, use local fallback
    return {
      text: this.getFallbackResponse(prompt, context),
      provider: "fallback",
      model: "local",
    };
  }

  async callProvider(provider, prompt, context) {
    const payload = this.formatPayload(provider, prompt, context);

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await fetch(provider.endpoint, {
          method: "POST",
          headers: provider.headers,
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(30000), // 30 second timeout
        });

        if (!response.ok) {
          if (response.status === 429) {
            // Rate limit - wait and retry
            await this.sleep(this.retryDelay * attempt);
            continue;
          }
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return this.extractResponse(provider, data);
      } catch (error) {
        if (attempt === this.maxRetries) {
          throw error;
        }
        await this.sleep(this.retryDelay * attempt);
      }
    }
  }

  formatPayload(provider, prompt, context) {
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
          max_tokens: context.maxTokens || 150,
          system: context.systemPrompt || "You are a helpful assistant.",
          messages: [{ role: "user", content: prompt }],
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

      default:
        throw new Error(`Unknown provider: ${provider.name}`);
    }
  }

  extractResponse(provider, data) {
    switch (provider.name) {
      case "openai":
        return data.choices?.[0]?.message?.content || "";

      // case 'anthropic':
      //   return data.content?.?.text || '';

      case "huggingface":
        if (Array.isArray(data) && data?.generated_text) {
          return data.generated_text.replace(data.inputs || "", "").trim();
        }
        return "";

      default:
        return "";
    }
  }

  getFallbackResponse(prompt, context) {
    // Enhanced local fallback based on context and prompt analysis
    const promptLower = prompt.toLowerCase();

    if (context.type === "pickup_line") {
      const templates = [
        `Just like Shah Rukh Khan in his movies, ${
          context.userInfo || "you"
        } have made my heart skip a beat!`,
        `If I were to write a love story, ${
          context.userInfo || "beautiful"
        }, you would be both the beginning and the happy ending.`,
        `Main hoon na, ${
          context.userInfo || "gorgeous"
        }? Because like SRK, I promise to always be there for you.`,
      ];
      return this.pickRandom(templates);
    }

    if (context.type === "flirty_reply") {
      if (promptLower.includes("hello") || promptLower.includes("hi")) {
        return `Namaste ${
          context.name || "beautiful"
        }! Like in my movies, you've made my heart do a little dance 💃`;
      }
      if (promptLower.includes("thank")) {
        return `Anything for you, ${
          context.name || "gorgeous"
        }! Like I always say in my films — main hoon na! 🤗`;
      }
      return `You always know just what to say to make me smile, ${
        context.name || "beautiful"
      }! 😊`;
    }

    // Generic fallback
    return "I'm here to help! Let me know what you'd like to know.";
  }

  pickRandom(array) {
    return array[Math.floor(Math.random() * array.length)];
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Health check for providers
  async checkProviderHealth() {
    const results = {};

    for (const provider of this.providers) {
      try {
        await this.callProvider(provider, "Test", { maxTokens: 1 });
        results[provider.name] = "healthy";
      } catch (error) {
        results[provider.name] = "unhealthy";
      }
    }

    return results;
  }
}

/* ──────────────────────────────────────────────────────────────
   3.  Utility helpers
   ────────────────────────────────────────────────────────────── */
const USER_AGENT = "Puch/1.0 (Autonomous)";
const DDG_HTML = "https://html.duckduckgo.com/html/?q=";
const llm = new LLMUtility();

/* Simple HTML → text extractor */
function stripHtml(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<\/?[a-z][^>]*>/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

/* Random chooser */
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

/* ──────────────────────────────────────────────────────────────
   4.  MCP Handler definition
   ────────────────────────────────────────────────────────────── */
const handlerFactory = (server) => {
  /* ----------------------------------------------------------
     4.1  validate  (required by Puch)
     ---------------------------------------------------------- */
  server.tool(
    "validate",
    "Returns the owner phone number in E.164-without-plus format",
    {},
    async () => ({
      content: [{ type: "text", text: MY_NUMBER }],
    })
  );

  /* ----------------------------------------------------------
     4.2  llm_health_check (diagnostic tool)
     ---------------------------------------------------------- */
  server.tool(
    "llm_health_check",
    "Check the health status of available LLM providers",
    {},
    async () => {
      const health = await llm.checkProviderHealth();
      const healthReport = Object.entries(health)
        .map(([provider, status]) => `• ${provider}: ${status}`)
        .join("\n");

      return {
        content: [
          {
            type: "text",
            text: `🔍 **LLM Provider Health Status**\n\n${healthReport}\n\n📊 Total providers: ${
              Object.keys(health).length
            }`,
          },
        ],
      };
    }
  );

  /* ----------------------------------------------------------
     4.3  generate_srk_pickup_line (Enhanced with LLM)
     ---------------------------------------------------------- */
  server.tool(
    "generate_srk_pickup_line",
    "Generate Shah Rukh Khan–style pickup lines using AI",
    {
      user_info: z.string().describe("Info about the user"),
      target_info: z
        .string()
        .optional()
        .describe("Info about the person to impress"),
      style: z
        .enum(["romantic", "witty", "bollywood", "classic"])
        .default("romantic")
        .describe("Style of pickup line"),
    },
    async ({ user_info, target_info, style }) => {
      const context = target_info
        ? `User: ${user_info}. Target: ${target_info}`
        : `User: ${user_info}`;

      const stylePrompts = {
        romantic: "Generate a deeply romantic and charming pickup line",
        witty: "Create a clever and witty pickup line with wordplay",
        bollywood: "Make a Bollywood-themed pickup line with movie references",
        classic: "Write a timeless, classic romantic pickup line",
      };

      const prompt = `${
        stylePrompts[style]
      } in Shah Rukh Khan's signature style.

Context: ${context}

The pickup line should be:
- Charming and romantic like SRK's famous dialogues
- ${style === "witty" ? "Clever and witty" : "Heartfelt and sincere"}
- Appropriate and respectful
- Include Bollywood flair and Hindi/Urdu phrases where appropriate
${target_info ? `- Personalized for the target person` : ""}

Generate only the pickup line:`;

      try {
        const result = await llm.generateText(prompt, {
          type: "pickup_line",
          userInfo: user_info,
          targetInfo: target_info,
          systemPrompt:
            "You are Shah Rukh Khan, the King of Bollywood romance. Generate charming pickup lines in your signature style.",
          maxTokens: 100,
          temperature: 0.8,
        });

        const providerInfo =
          result.provider !== "fallback"
            ? `\n\n🤖 *Generated using ${result.provider} (${result.model})*`
            : `\n\n_(Using curated SRK-style fallback)_`;

        return {
          content: [
            {
              type: "text",
              text: `💕 **Shah Rukh Khan-style Pickup Line** (${style}) 💕\n\n*"${result.text}"*\n\n✨ *Delivered with SRK's signature charm!*${providerInfo}`,
            },
          ],
        };
      } catch (error) {
        // Ultimate fallback
        const fallbackLines = [
          `Just like in my movies, ${user_info}, you've made my heart skip a beat. Kuch Kuch Hota Hai when I see you!`,
          `If I were to write a love story, ${user_info}, you would be both the beginning and the happy ending.`,
          `Main hoon na, ${user_info}? Because like SRK, I promise to always be there for you.`,
        ];

        return {
          content: [
            {
              type: "text",
              text: `💕 **Shah Rukh Khan-style Pickup Line** 💕\n\n*"${pick(
                fallbackLines
              )}"*\n\n✨ *Delivered with SRK's signature charm!*\n\n_(Emergency fallback mode)_`,
            },
          ],
        };
      }
    }
  );

  /* ----------------------------------------------------------
     4.4  find_date_locations
     ---------------------------------------------------------- */
  server.tool(
    "find_date_locations",
    "Suggest romantic date spots via web search",
    {
      city: z.string().describe("City or locality"),
      date_type: z.string().default("romantic").describe("Type of date"),
      budget: z
        .string()
        .default("moderate")
        .describe("Budget: budget-friendly|moderate|upscale"),
    },
    async ({ city, date_type, budget }) => {
      try {
        const query = `best ${date_type} date spots ${city} ${budget} restaurants cafes`;
        const res = await fetch(`${DDG_HTML}${encodeURIComponent(query)}`, {
          headers: { "User-Agent": USER_AGENT },
          signal: AbortSignal.timeout(10000), // 10 second timeout
        });

        if (!res.ok) {
          throw new Error(`Search failed: ${res.statusText}`);
        }

        const html = await res.text();
        const matches = [
          ...html.matchAll(
            /result__a[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>[\s\S]*?result__snippet">(.*?)<\/a>/gi
          ),
        ];
        const locations = matches
          .slice(0, 8)
          .map(([, url, title, snippet]) => ({
            url,
            title: stripHtml(title),
            snippet: stripHtml(snippet).slice(0, 200),
          }));

        const header = `💝 **Date Locations in ${city}** 💝\n\n**Date Type:** ${date_type}\n**Budget:** ${budget}\n\n`;

        if (!locations.length) {
          return {
            content: [
              {
                type: "text",
                text: `${header}❌ No specific spots found.\n\n• Try well-rated local cafés\n• Scenic parks\n• Art museums\n• Cozy rooftop lounges`,
              },
            ],
          };
        }

        const body = locations
          .map(
            (loc, i) =>
              `**${i + 1}. ${loc.title}**\n ${loc.snippet}…\n 🔗 <${loc.url}>\n`
          )
          .join("\n");

        const tips =
          "💡 **Pro Tips:**\n• Reserve ahead for popular venues\n• Check opening hours\n• Keep indoor options for bad weather";

        return {
          content: [{ type: "text", text: `${header}${body}\n${tips}` }],
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `❌ **Error finding date locations:** ${error.message}\n\n💡 **General suggestions for ${date_type} dates in ${city}:**\n• Romantic restaurants\n• Cozy cafes\n• Scenic parks\n• Local attractions\n• Entertainment venues`,
            },
          ],
        };
      }
    }
  );

  /* ----------------------------------------------------------
     4.5  generate_srk_flirty_reply (Enhanced with LLM)
     ---------------------------------------------------------- */
  server.tool(
    "generate_srk_flirty_reply",
    "Craft flirty SRK-style replies using AI",
    {
      message: z.string().describe("Incoming message to reply to"),
      your_name: z
        .string()
        .optional()
        .describe("Your name (for personalisation)"),
      context: z.string().optional().describe("Conversation context"),
      tone: z
        .enum(["flirty", "romantic", "witty", "sweet"])
        .default("flirty")
        .describe("Tone of the reply"),
    },
    async ({ message, your_name, context, tone }) => {
      const name = your_name || "beautiful";

      const tonePrompts = {
        flirty: "Generate a playfully flirty and charming reply",
        romantic: "Create a deeply romantic and heartfelt response",
        witty: "Write a clever and witty reply with humor",
        sweet: "Make a sweet and endearing response",
      };

      const prompt = `${
        tonePrompts[tone]
      } to this message in Shah Rukh Khan's signature style.

Message to reply to: "${message}"
${context ? `Conversation context: ${context}` : ""}
${your_name ? `Your name: ${your_name}` : ""}

The reply should be:
- ${tone} and engaging like SRK's famous dialogues
- Charming with Bollywood flair
- Conversation-continuing and appropriate
- Include Hindi/Urdu phrases where suitable
- Sound like something SRK would say in his romantic movies

Generate only the SRK-style reply:`;

      try {
        const result = await llm.generateText(prompt, {
          type: "flirty_reply",
          name: your_name,
          originalMessage: message,
          systemPrompt:
            "You are Shah Rukh Khan responding to messages with your signature romantic charm and Bollywood flair.",
          maxTokens: 120,
          temperature: 0.8,
        });

        const providerInfo =
          result.provider !== "fallback"
            ? `\n\n🤖 *Generated using ${result.provider} (${result.model})*`
            : `\n\n_(Using curated SRK-style responses)_`;

        return {
          content: [
            {
              type: "text",
              text: `💕 **SRK-Style ${
                tone.charAt(0).toUpperCase() + tone.slice(1)
              } Reply** 💕\n\n**Original Message:** "${message}"\n\n**Your Shah Rukh Khan Response:**\n*"${
                result.text
              }"*\n\n✨ *Delivered with SRK's signature charm and Bollywood romance!*${providerInfo}`,
            },
          ],
        };
      } catch (error) {
        // Fallback classification and response
        const lowercase = message.toLowerCase();

        let fallbackReply;
        if (/hello|hi|hey|morning|evening/.test(lowercase)) {
          fallbackReply = `Namaste ${name}! Like in my movies, you've made my heart do a little dance 💃`;
        } else if (/thank|thanks/.test(lowercase)) {
          fallbackReply = `Anything for you, ${name}! Like I always say in my films — main hoon na! 🤗`;
        } else if (/sorry|apologize/.test(lowercase)) {
          fallbackReply = `Don't worry, ${name}! True love means never having to say sorry 💕`;
        } else {
          fallbackReply = `You always know just what to say to make me smile, ${name}! It's like you have the perfect dialogue for every scene 😊`;
        }

        return {
          content: [
            {
              type: "text",
              text: `💕 **SRK-Style Flirty Reply** 💕\n\n**Original Message:** "${message}"\n\n**Your Shah Rukh Khan Response:**\n*"${fallbackReply}"*\n\n✨ *Delivered with SRK's signature charm and Bollywood romance!*\n\n_(Emergency fallback mode)_`,
            },
          ],
        };
      }
    }
  );
};

/* ──────────────────────────────────────────────────────────────
   5.  Export (Vercel/Cloudflare compatible)
   ────────────────────────────────────────────────────────────── */
export const { GET, POST } = createMcpHandler(
  handlerFactory,
  { basePath: "/api", auth: { type: "bearer", token: TOKEN } },
  { verboseLogs: true }
);

// Add this to avoid 405 on preflight and other OPTIONS requests
export async function OPTIONS(request) {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Authorization, Content-Type",
    },
  });
}
