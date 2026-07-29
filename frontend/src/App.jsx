import { useState } from "react";
import { Outlet } from "react-router-dom";
import SessionSidebar from "./components/SessionSidebar.jsx";

export default function App() {
  // Mobile: sidebar starts closed (it would otherwise cover the whole
  // screen on small viewports, leaving no room for the debate form/view).
  // Desktop (md breakpoint+): sidebar is always visible regardless of
  // this state — see the `md:translate-x-0` override below.
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen flex">
      {/* Overlay backdrop — only rendered on mobile when the sidebar is
          open, so tapping outside it closes the drawer. */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-30
          w-72 shrink-0 border-r border-slate-200 bg-white
          transform transition-transform duration-200 ease-in-out
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          md:translate-x-0
        `}
      >
        <SessionSidebar onClose={() => setSidebarOpen(false)} />
      </aside>

      <main className="flex-1 min-w-0 p-6">
        {/* Hamburger — only visible on mobile (md:hidden), opens the
            sidebar drawer. Hidden once the drawer is already open since
            the sidebar's own close button (X) takes over at that point. */}
        {!sidebarOpen && (
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
            className="md:hidden mb-4 inline-flex items-center justify-center w-9 h-9 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
          >
            {/* Hamburger icon (text-only, no icon-font dependency) */}
            <span className="text-lg leading-none">☰</span>
          </button>
        )}
        <Outlet />
      </main>
    </div>
  );
}