# HazeTrace app refresh — design decisions (multi-agent debate, 14 Nov 2026)

Four reviewers proposed features independently (NASA Space Apps judge, stakeholder/SDG specialist,
UI/UX designer, feasibility engineer), then voted on the open questions. Final decisions:

| # | Question | Decision | Vote |
|---|---|---|---|
| 1 | Map library | Plotly `choropleth_map` with true 0.25° square cells (GeoJSON), same library for map and time-lapse; fallback `scatter_map` | 3 B → engineer proposed C (squares in Plotly), designer's square argument kept |
| 2 | Colour ramps | Cividis on the calendar (accessibility), trimmed inferno on the dark map | 3–1 |
| 3 | Bulletin | "HazeTrace Data Note", levels **Typical / Above usual / Well above usual / Exceptional**, ≥ 50 hotspots before any level above Typical; never "alert", "warning", "live", "extreme" | 4–0 (wording from stakeholder) |
| 4 | Stakeholders | Static persona cards with the question each answers, not a persona switch | 4–0 (judge changed mind) |
| 5 | SDGs | 13.1, 13.3, 15.1, 15.3 (interpretation), 17.18. Health (3.9) and urban air (11.6) only as motivation, never claimed | 4–0 (judge dropped 3.9) |
| 6 | Partial Nov 2026 | KPIs and data note use Oct 2026 (latest complete month). Nov shown on the map only, labelled "partial to 13 Nov, NRT", never ranked | 4–0 |
| 7 | Time-lapse | Default: one frame per year, Aug–Oct season total, 2003–2026. Toggle: months of a chosen year | 4–0 |

Also agreed: dark smoke/ember theme; KPI strip ≤ 5 cards; ML/trust visible early (validation scores, NOAA-20
transfer, honest Malaysia check, varying-ratio finding); transboundary view with neutral wording
(regions, not countries or companies; no smoke-transport claim); no push notifications, forecasts,
health advice, drawn AOI or 3D; "latest data 13 Nov 2026" instead of "live".

Note: the numbers quoted during the debate were later superseded. Current results are in README.md (Results).
