import "@fontsource/archivo/600.css";
import "@fontsource/archivo/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./index.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RunProvider } from "./lib/run";
import { Compare } from "./pages/Compare";
import { Dashboard } from "./pages/Dashboard";
import { Data } from "./pages/Data";
import { Live } from "./pages/Live";
import { MachineDetail } from "./pages/MachineDetail";
import { Machines } from "./pages/Machines";
import { Scenarios } from "./pages/Scenarios";
import { Stock } from "./pages/Stock";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RunProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="direct" element={<Live />} />
              <Route path="machines" element={<Machines />} />
              <Route path="machines/:id" element={<MachineDetail />} />
              <Route path="stock" element={<Stock />} />
              <Route path="comparaison" element={<Compare />} />
              <Route path="scenarios" element={<Scenarios />} />
              <Route path="donnees" element={<Data />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </RunProvider>
    </QueryClientProvider>
  </StrictMode>,
);
