import clsx from "clsx";
import type { ButtonHTMLAttributes, ReactNode, SelectHTMLAttributes, InputHTMLAttributes } from "react";

export function Page({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-[1320px] px-8 pt-10 pb-16">
      <header className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="max-w-[62ch]">
          <h1 className="text-[28px] leading-tight font-semibold tracking-[-0.01em]">{title}</h1>
          {description && <p className="mt-2 text-ink-soft">{description}</p>}
        </div>
        {actions}
      </header>
      <div className="mt-8">{children}</div>
    </div>
  );
}

export function Section({
  title,
  description,
  aside,
  children,
  className,
}: {
  title: string;
  description?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <h2 className="text-[18px] font-semibold">{title}</h2>
          {description && <p className="mt-0.5 text-[13px] text-ink-soft">{description}</p>}
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "quiet";
};

export function Button({ variant = "primary", className, ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      className={clsx(
        "inline-flex h-10 items-center justify-center gap-2 rounded-sm px-4 text-[14px] font-medium transition-colors duration-150 ease-out disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-plum-600 text-white hover:bg-plum-700",
        variant === "secondary" && "border border-line-strong bg-paper text-ink hover:bg-mist",
        variant === "quiet" && "text-plum-700 hover:bg-plum-50",
        className,
      )}
    />
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[13px] font-medium text-ink-soft">{label}</span>
      {children}
      {hint && <span className="text-[12px] text-ink-soft">{hint}</span>}
    </label>
  );
}

const controlClass =
  "h-10 rounded-sm border border-line-strong bg-paper px-3 text-[14px] text-ink transition-colors duration-150 ease-out hover:border-plum-400 focus:border-plum-600";

export function Select({ className, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...rest} className={clsx(controlClass, "pr-8", className)} />;
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...rest} className={clsx(controlClass, "tabular", className)} />;
}

export function Notice({ tone = "info", children }: { tone?: "info" | "error"; children: ReactNode }) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={clsx(
        "rounded-md px-4 py-3 text-[14px]",
        tone === "error" ? "bg-danger-soft text-danger" : "bg-sun-100 text-sun-800",
      )}
    >
      {children}
    </div>
  );
}

export type KpiItem = { label: string; value: string; hint?: string; emphasis?: boolean };

/** Liste d'indicateurs : libellé à gauche, valeur à droite, en chiffres tabulaires. */
export function KpiList({ items }: { items: KpiItem[] }) {
  return (
    <dl className="divide-y divide-line">
      {items.map((item) => (
        <div key={item.label} className="flex items-baseline justify-between gap-4 py-2.5">
          <dt className="text-[14px] text-ink-soft">
            {item.label}
            {item.hint && <span className="block text-[12px] text-ink-soft/90">{item.hint}</span>}
          </dt>
          <dd
            className={clsx(
              "tabular text-right font-medium text-ink",
              item.emphasis ? "text-[22px] font-medium" : "text-[15px]",
            )}
          >
            {item.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Loading({ label }: { label: string }) {
  return <p className="py-10 text-ink-soft">{label}</p>;
}
