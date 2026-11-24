// Tipos del dominio «auth» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

// Auth types
export type UserRole = "user" | "model_manager" | "system_admin";

export interface AuthUser {
  id: string;
  username: string;
  email: string | null;
  role: UserRole;
  createdAt: string;
}

export interface ApiKey {
  id: string;
  name: string;
  keyPrefix: string;
  expiresAt: string | null;
  lastUsedAt: string | null;
  isRevoked: boolean;
  createdAt: string;
  groupId?: string | null;
}

// An API key as seen by an admin in the "all keys" view, including its owner.
export interface AdminApiKey extends ApiKey {
  userId: string;
  username: string;
}

export interface Group {
  id: string;
  name: string;
  description: string;
  isDefault: boolean;
  grantsAllModels: boolean;
  memberCount: number;
  modelCount: number;
  createdAt: string;
}

// Admin user management
export interface AdminUser {
  id: string;
  username: string;
  email: string | null;
  role: UserRole;
  isActive: boolean;
  createdAt: string;
}

export interface GroupMember {
  userId: string;
  username: string;
  role: UserRole;
}

export interface CreateUserPayload {
  username: string;
  password: string;
  role: UserRole;
  email?: string | null;
}

export interface UpdateUserPayload {
  role?: UserRole;
  isActive?: boolean;
  email?: string | null;
}

export interface CreateGroupPayload {
  name: string;
  description?: string | null;
  grantsAllModels?: boolean;
}
