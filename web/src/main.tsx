import "@fontsource/archivo/600.css";
import "@fontsource/archivo/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./index.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RunProvider } from "./lib/run";
import { Live } from "./pages/Live";
import { Machines } from "./pages/Machines";
import { Scenarios } from "./pages/Scenarios";

// Pages lourdes (graphiques ECharts) chargées à la demande : le moteur de rendu
// n'est téléchargé que par les vues qui l'utilisent, pas au démarrage.
const Dashboard = lazy(() => import("./pages/Dashboard").then((m) => ({ default: m.Dashboard })));
const Compare = lazy(() => import("./pages/Compare").then((m) => ({ default: m.Compare })));
const Data = lazy(() => import("./pages/Data").then((m) => ({ default: m.Data })));
const Evaluation = lazy(() => import("./pages/Evaluation").then((m) => ({ default: m.Evaluation })));
const MachineDetail = lazy(() =>
  import("./pages/MachineDetail").then((m) => ({ default: m.MachineDetail })),
);
const Reproductibilite = lazy(() =>
  import("./pages/Reproductibilite").then((m) => ({ default: m.Reproductibilite })),
);
const Stock = lazy(() => import("./pages/Stock").then((m) => ({ default: m.Stock })));

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
              <Route path="evaluation" element={<Evaluation />} />
              <Route path="scenarios" element={<Scenarios />} />
              <Route path="donnees" element={<Data />} />
              <Route path="reproductibilite" element={<Reproductibilite />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </RunProvider>
    </QueryClientProvider>
  </StrictMode>,
);
