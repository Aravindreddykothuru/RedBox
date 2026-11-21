import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// Only created when the project is configured; otherwise the app runs in demo mode (see session.ts).
export const supabase: SupabaseClient | null =
  url && key && typeof window !== "undefined"
    ? createClient(url, key, { auth: { persistSession: true, autoRefreshToken: true } })
    : null;
