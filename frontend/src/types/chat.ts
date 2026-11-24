// Tipos del dominio «chat» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

export type ChatAudioFormat = "wav" | "mp3" | "m4a" | "webm" | "ogg" | "flac";

export interface ChatTextPart {
  type: "text";
  text: string;
}

export interface ChatImageUrlPart {
  type: "image_url";
  image_url: { url: string };
}

export interface ChatInputAudioPart {
  type: "input_audio";
  input_audio: { data: string; format: ChatAudioFormat };
}

export type ChatContentPart =
  ChatTextPart | ChatImageUrlPart | ChatInputAudioPart;

export interface ChatRequestMessage {
  role: "system" | "user" | "assistant";
  content: string | ChatContentPart[];
}
