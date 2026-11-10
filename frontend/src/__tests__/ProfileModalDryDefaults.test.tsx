import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LLMDefaults } from "@/components/models/ProfileModal";

describe("LLMDefaults DRY fields", () => {
  it("adds dry_multiplier without dropping existing defaults", () => {
    const onChange = vi.fn();
    render(
      <LLMDefaults
        defaults={{ temperature: 0, repeat_penalty: 1.15 }}
        onChange={onChange}
      />,
    );

    const input = screen.getByText("DRY multiplicador").parentElement!.querySelector("input")!;
    fireEvent.change(input, { target: { value: "0.8" } });

    expect(onChange).toHaveBeenLastCalledWith({
      temperature: 0,
      repeat_penalty: 1.15,
      dry_multiplier: 0.8,
    });
  });

  it("shows stored DRY values and clears them when emptied", () => {
    const onChange = vi.fn();
    render(
      <LLMDefaults
        defaults={{ dry_multiplier: 0.8, dry_penalty_last_n: 4096 }}
        onChange={onChange}
      />,
    );

    const windowInput = screen
      .getByText("DRY ventana (tokens)")
      .parentElement!.querySelector("input")!;
    expect(windowInput.value).toBe("4096");

    fireEvent.change(windowInput, { target: { value: "" } });
    expect(onChange).toHaveBeenLastCalledWith({ dry_multiplier: 0.8 });
  });
});
