// LLM provider brand marks for the header's model-provider dropdown.
// Unlike Icons.tsx's monochrome currentColor strokes (meant for generic UI
// chrome), these are each provider's own distinct logo shape/color — a
// provider selector needs to actually look like the provider, not blend into
// the surrounding button.

interface ProviderIconProps {
  size?: number;
  className?: string;
}

export function IconOpenAI({ size = 16, className }: ProviderIconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        fill="#10A37F"
        d="M22.28 9.82a5.99 5.99 0 0 0-.52-4.91 6.05 6.05 0 0 0-6.51-2.9A6.02 6.02 0 0 0 4.98 4.18a5.99 5.99 0 0 0-4 2.9 6.05 6.05 0 0 0 .75 7.1 5.99 5.99 0 0 0 .52 4.91 6.05 6.05 0 0 0 6.51 2.9 5.99 5.99 0 0 0 4.51 2.01 6.05 6.05 0 0 0 5.77-4.19 5.99 5.99 0 0 0 4-2.9 6.05 6.05 0 0 0-.76-7.09Zm-9.02 12.62a4.49 4.49 0 0 1-2.89-1.04l.14-.08 4.8-2.77a.78.78 0 0 0 .4-.68v-6.77l2.03 1.17a.07.07 0 0 1 .04.06v5.6a4.5 4.5 0 0 1-4.52 4.51ZM3.6 18.36a4.48 4.48 0 0 1-.54-3.02l.14.09 4.8 2.77a.78.78 0 0 0 .78 0l5.86-3.38v2.34a.08.08 0 0 1-.03.07l-4.85 2.8a4.5 4.5 0 0 1-6.16-1.67ZM2.34 7.9A4.48 4.48 0 0 1 4.69 5.9v5.69a.78.78 0 0 0 .39.68l5.86 3.38-2.03 1.17a.07.07 0 0 1-.07 0L3.99 14a4.5 4.5 0 0 1-1.65-6.1Zm16.66 3.88-5.86-3.39L15.17 7.2a.07.07 0 0 1 .07 0l4.85 2.81a4.5 4.5 0 0 1-.68 8.11v-5.7a.78.78 0 0 0-.4-.67Zm2.02-3.04-.14-.09-4.8-2.77a.78.78 0 0 0-.78 0l-5.86 3.38V6.94a.07.07 0 0 1 .03-.07l4.85-2.8a4.5 4.5 0 0 1 6.7 4.67ZM9.03 12.87l-2.03-1.17a.08.08 0 0 1-.04-.06v-5.6a4.5 4.5 0 0 1 7.38-3.46l-.14.08-4.8 2.77a.78.78 0 0 0-.4.68l.03 6.76Zm1.1-2.38 2.61-1.51 2.61 1.5v3.02l-2.61 1.51-2.61-1.5v-3.02Z"
      />
    </svg>
  );
}

export function IconAnthropic({ size = 16, className }: ProviderIconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        fill="#D97757"
        d="M17.3 4h-3.65l6.13 16h3.65L17.3 4Zm-10.6 0L.57 20h3.72l1.25-3.28h6.46L13.25 20h3.72L10.85 4H6.7Zm-.03 9.87 2.12-5.59 2.12 5.59H6.67Z"
      />
    </svg>
  );
}

export function IconGemini({ size = 16, className }: ProviderIconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="gemini-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#4285F4" />
          <stop offset="50%" stopColor="#9B72CB" />
          <stop offset="100%" stopColor="#D96570" />
        </linearGradient>
      </defs>
      <path
        fill="url(#gemini-grad)"
        d="M12 2c0 5.52 4.48 10 10 10-5.52 0-10 4.48-10 10 0-5.52-4.48-10-10-10 5.52 0 10-4.48 10-10Z"
      />
    </svg>
  );
}

export function IconDeepSeek({ size = 16, className }: ProviderIconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path
        fill="#4D6BFE"
        d="M21.5 6.2c-.4-.2-.6.2-.85.4-.9.75-1.95 1.05-3.15.9-1.7-.2-3.15.35-4.35 1.65-.25-1.2-.9-2.1-1.9-2.75-.5-.3-1-.55-1.35-1.05-.25-.35-.3-.75-.4-1.15-.05-.2-.1-.4-.35-.45-.25-.05-.4.15-.55.3-.55.65-.75 1.45-.7 2.3.05 1.7.75 3.05 2.15 4.05.15.1.3.2.3.45-.05.4-.35.6-.65.75-.55.3-1.15.5-1.55 1.05-.45-.35-.9-.45-1.4-.35-.85.15-1.55.6-2.15 1.2-.9.9-1.35 2-1.35 3.25 0 .3.1.5.4.55.35.05.6-.15.85-.35.4-.35.85-.6 1.4-.65.1 1.1.55 2 1.35 2.75 1.05.95 2.3 1.35 3.7 1.15.9-.15 1.7-.55 2.4-1.15.15.85.6 1.35 1.45 1.55.9.2 1.75-.05 2.4-.7.3-.3.5-.7.5-1.15 0-.3-.15-.5-.4-.6-.3-.1-.55.05-.75.3-.2.25-.45.35-.75.25-.35-.1-.5-.4-.5-.75-.05-.7.3-1.2.85-1.6.85-.6 1.35-1.4 1.55-2.4.05-.3.05-.6.05-.9.7.05 1.3.35 1.85.75.25.2.5.4.85.35.35-.05.5-.35.45-.7-.15-1.15-.7-2.05-1.7-2.65-.6-.35-1.25-.5-1.95-.4.4-.65.5-1.35.35-2.1-.15-.75-.55-1.35-1.15-1.8.5-.3.9-.7 1.15-1.2.35-.65.4-1.35.15-2.05-.1-.25-.25-.45-.55-.6Z"
      />
    </svg>
  );
}
