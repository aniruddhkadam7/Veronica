'use client';

import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';

export const observeActivity = (
  el: Element,
  onChange: (active: boolean) => void,
): (() => void) => {
  // Deliberately does NOT also gate on `document.visibilityState`/
  // `visibilitychange`: an always-on-top, never-focused host window (e.g.
  // Veronica's floating widget, see veronica_widget.rs) is fully visible to
  // the user but can report `visibilityState: "hidden"` in some WebView2
  // configurations because it never receives OS focus. Gating the orb's
  // animation loop on that flag silently froze the widget's orb — it never
  // showed the listening/thinking/speaking transitions even though the
  // pipeline state was updating correctly. IntersectionObserver alone still
  // correctly pauses the animation when the orb's own element is actually
  // scrolled out of view or its document is torn down.
  let active = true;

  const observer = new IntersectionObserver((entries) => {
    const inView = entries[entries.length - 1]?.isIntersecting ?? true;
    if (inView === active) return;
    active = inView;
    onChange(inView);
  });
  observer.observe(el);

  return () => {
    observer.disconnect();
  };
};

export const useInView = (
  ref: RefObject<Element | null>,
  onChange?: (active: boolean) => void,
): RefObject<boolean> => {
  const activeRef = useRef(true);
  const onChangeRef = useRef(onChange);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const unobserve = observeActivity(el, (active) => {
      activeRef.current = active;
      onChangeRef.current?.(active);
    });
    return () => {
      unobserve();
      activeRef.current = true;
    };
  }, [ref]);

  return activeRef;
};
