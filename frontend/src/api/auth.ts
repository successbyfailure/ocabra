// Cliente tipado de /ocabra/* — dominio «auth» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { request } from "@/api/http"
import { isRecord } from "@/api/_helpers"
import type {
  AdminApiKey,
  AdminUser,
  ApiKey,
  AuthUser,
  CreateGroupPayload,
  CreateUserPayload,
  Group,
  GroupMember,
  UpdateUserPayload,
} from "@/types"

export function toAuthUser(raw: unknown): AuthUser {
  const d = isRecord(raw) ? raw : {};
  return {
    id: String(d.id ?? ""),
    username: String(d.username ?? ""),
    email: (d.email ?? null) as string | null,
    role: String(d.role ?? "user") as AuthUser["role"],
    createdAt: String(d.created_at ?? d.createdAt ?? ""),
  };
}


export function toAdminUser(raw: unknown): AdminUser {
  const d = isRecord(raw) ? raw : {};
  return {
    id: String(d.id ?? ""),
    username: String(d.username ?? ""),
    email: (d.email ?? null) as string | null,
    role: String(d.role ?? "user") as AdminUser["role"],
    isActive: Boolean(d.is_active ?? d.isActive ?? true),
    createdAt: String(d.created_at ?? d.createdAt ?? ""),
  };
}


export function toGroupMember(raw: unknown): GroupMember {
  const d = isRecord(raw) ? raw : {};
  return {
    userId: String(d.user_id ?? d.userId ?? d.id ?? ""),
    username: String(d.username ?? ""),
    role: String(d.role ?? "user") as GroupMember["role"],
  };
}


export function toGroup(raw: unknown): Group {
  const d = isRecord(raw) ? raw : {};
  return {
    id: String(d.id ?? ""),
    name: String(d.name ?? ""),
    description: String(d.description ?? ""),
    isDefault: Boolean(d.is_default ?? d.isDefault ?? false),
    grantsAllModels: Boolean(d.grants_all_models ?? d.grantsAllModels ?? false),
    memberCount: Number(d.member_count ?? d.memberCount ?? 0),
    modelCount: Number(d.model_count ?? d.modelCount ?? 0),
    createdAt: String(d.created_at ?? d.createdAt ?? ""),
  };
}


export function toApiKey(raw: unknown): ApiKey {
  const d = isRecord(raw) ? raw : {};
  return {
    id: String(d.id ?? ""),
    name: String(d.name ?? ""),
    keyPrefix: String(d.key_prefix ?? d.keyPrefix ?? ""),
    expiresAt: (d.expires_at ?? d.expiresAt ?? null) as string | null,
    lastUsedAt: (d.last_used_at ?? d.lastUsedAt ?? null) as string | null,
    isRevoked: Boolean(d.is_revoked ?? d.isRevoked ?? false),
    createdAt: String(d.created_at ?? d.createdAt ?? ""),
  };
}


export function toAdminApiKey(raw: unknown): AdminApiKey {
  const d = isRecord(raw) ? raw : {};
  return {
    ...toApiKey(raw),
    userId: String(d.user_id ?? d.userId ?? ""),
    username: String(d.username ?? ""),
  };
}


export const authApiModule = {
  auth: {
    login: async (
      username: string,
      password: string,
      remember?: boolean,
    ): Promise<{ user: AuthUser }> => {
      const raw = await request<Record<string, unknown>>(
        "POST",
        "/ocabra/auth/login",
        {
          username,
          password,
          remember: remember ?? false,
        },
      );
      return { user: toAuthUser(raw.user ?? raw) };
    },

    logout: (): Promise<void> => request<void>("POST", "/ocabra/auth/logout"),

    me: async (): Promise<AuthUser> =>
      toAuthUser(await request<unknown>("GET", "/ocabra/auth/me")),

    listKeys: async (): Promise<ApiKey[]> =>
      (await request<unknown[]>("GET", "/ocabra/auth/keys")).map(toApiKey),

    createKey: async (
      name: string,
      expiresInDays?: number | null,
    ): Promise<ApiKey & { key: string }> => {
      const raw = await request<Record<string, unknown>>(
        "POST",
        "/ocabra/auth/keys",
        {
          name,
          expires_in_days: expiresInDays ?? null,
        },
      );
      return { ...toApiKey(raw), key: String(raw.key ?? "") };
    },

    revokeKey: (keyId: string): Promise<void> =>
      request<void>("DELETE", `/ocabra/auth/keys/${encodeURIComponent(keyId)}`),

    purgeRevokedKeys: (): Promise<{ deleted: number }> =>
      request("DELETE", "/ocabra/auth/keys/revoked"),
  },

  users: {
    list: async (): Promise<AdminUser[]> =>
      (await request<unknown[]>("GET", "/ocabra/users")).map(toAdminUser),
    create: async (data: CreateUserPayload): Promise<AdminUser> =>
      toAdminUser(
        await request<unknown>("POST", "/ocabra/users", {
          username: data.username,
          password: data.password,
          role: data.role,
          email: data.email ?? null,
        }),
      ),
    update: async (id: string, data: UpdateUserPayload): Promise<AdminUser> =>
      toAdminUser(
        await request<unknown>(
          "PATCH",
          `/ocabra/users/${encodeURIComponent(id)}`,
          {
            role: data.role,
            is_active: data.isActive,
            email: data.email,
          },
        ),
      ),
    remove: (id: string): Promise<void> =>
      request<void>("DELETE", `/ocabra/users/${encodeURIComponent(id)}`),
    resetPassword: (id: string, newPassword: string): Promise<void> =>
      request<void>(
        "POST",
        `/ocabra/users/${encodeURIComponent(id)}/reset-password`,
        {
          new_password: newPassword,
        },
      ),
    listKeys: async (id: string): Promise<ApiKey[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/users/${encodeURIComponent(id)}/keys`,
        )
      ).map(toApiKey),
    revokeKey: (userId: string, keyId: string): Promise<void> =>
      request<void>(
        "DELETE",
        `/ocabra/users/${encodeURIComponent(userId)}/keys/${encodeURIComponent(keyId)}`,
      ),
    createKeyForUser: async (
      userId: string,
      payload: {
        name: string;
        expiresInDays?: number | null;
        groupId?: string | null;
      },
    ) =>
      request<{
        id: string;
        name: string;
        keyPrefix: string;
        key: string;
        expiresAt: string | null;
        createdAt: string;
        groupId: string | null;
      }>("POST", `/ocabra/users/${encodeURIComponent(userId)}/keys`, {
        name: payload.name,
        expires_in_days: payload.expiresInDays ?? null,
        group_id: payload.groupId ?? null,
      }),
    // Admin "all keys" view + bulk reassignment.
    listAllKeys: async (): Promise<AdminApiKey[]> =>
      (await request<unknown[]>("GET", "/ocabra/api-keys")).map(toAdminApiKey),
    reassignKeys: (
      keyIds: string[],
      targetUserId: string,
    ): Promise<{
      reassigned: number;
      skipped: number;
      failed: Array<{ id: string; error: string }>;
    }> =>
      request("POST", "/ocabra/api-keys/reassign", {
        key_ids: keyIds,
        target_user_id: targetUserId,
      }),
    purgeRevokedKeys: (): Promise<{ deleted: number }> =>
      request("DELETE", "/ocabra/api-keys/revoked"),
  },

  groups: {
    list: async (): Promise<Group[]> =>
      (await request<unknown[]>("GET", "/ocabra/groups")).map(toGroup),
    create: async (data: CreateGroupPayload): Promise<Group> =>
      toGroup(
        await request<unknown>("POST", "/ocabra/groups", {
          name: data.name,
          description: data.description ?? "",
          grants_all_models: data.grantsAllModels ?? false,
        }),
      ),
    remove: (id: string): Promise<void> =>
      request<void>("DELETE", `/ocabra/groups/${encodeURIComponent(id)}`),
    listModels: async (id: string): Promise<string[]> => {
      const raw = await request<unknown[]>(
        "GET",
        `/ocabra/groups/${encodeURIComponent(id)}/models`,
      );
      return raw.map((item) => {
        if (typeof item === "string") return item;
        const r = isRecord(item) ? item : {};
        return String(r.model_id ?? r.modelId ?? r.id ?? item);
      });
    },
    addModel: (id: string, modelId: string): Promise<void> =>
      request<void>("POST", `/ocabra/groups/${encodeURIComponent(id)}/models`, {
        model_id: modelId,
      }),
    removeModel: (id: string, modelId: string): Promise<void> =>
      request<void>(
        "DELETE",
        `/ocabra/groups/${encodeURIComponent(id)}/models/${encodeURIComponent(modelId)}`,
      ),
    listMembers: async (id: string): Promise<GroupMember[]> =>
      (
        await request<unknown[]>(
          "GET",
          `/ocabra/groups/${encodeURIComponent(id)}/members`,
        )
      ).map(toGroupMember),
    addMember: (id: string, userId: string): Promise<void> =>
      request<void>(
        "POST",
        `/ocabra/groups/${encodeURIComponent(id)}/members`,
        {
          user_id: userId,
        },
      ),
    removeMember: (id: string, userId: string): Promise<void> =>
      request<void>(
        "DELETE",
        `/ocabra/groups/${encodeURIComponent(id)}/members/${encodeURIComponent(userId)}`,
      ),
  },

};
