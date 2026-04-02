# Estado: Stream 3-A — OpenAI API

Estado: completado.

Área principal:
- `backend/ocabra/api/openai/`

Resultado actual:
- compatibilidad `/v1/*` estabilizada
- alias `backend_model_id` aceptado además de `model_id` canónico
- stats y load-on-demand integrados
- streaming y envelopes de error alineados con comportamiento real

Pendiente real:
- ampliar e2e en CI cuando el entorno de tests backend completo esté disponible

Referencia viva:
- `docs/PLAN.md`
- `docs/CONTRACTS.md`
