# Arquitectura actual

## Objetivo

`retail-recommender-mlops` es un sistema MLOps local para construir, servir y
monitorear recomendaciones de productos sobre el dataset Instacart Market
Basket Analysis.

La versión actual prioriza un flujo completo y reproducible en local: datos,
entrenamiento, artifact, API, persistencia operativa, métricas y ejecución con
Docker Compose.

## Componentes principales

| Componente | Responsabilidad |
| --- | --- |
| `data/raw/instacart/` | CSV originales de Instacart y catálogo `products.csv`. |
| `ml/training/` | Análisis, preparación del dataset compacto y entrenamiento item-item. |
| `data/processed/training_baskets_v1_compact.parquet` | Dataset procesado usado por el entrenamiento actual. |
| `artifacts/models/item_item_recommender_v1.parquet` | Artifact entrenado con recomendaciones precalculadas. |
| `ml/serving/local_recommender.py` | Carga el artifact y genera recomendaciones para un carrito. |
| `backend/app/` | API FastAPI, schemas, servicios, conexión a PostgreSQL y rutas. |
| PostgreSQL | Guarda logs operativos en `recommendation_logs`. |
| `infra/docker/docker-compose.yml` | Levanta PostgreSQL y el backend para desarrollo local. |

El backend carga el recomendador una sola vez durante el arranque. El artifact
y el catálogo se montan como volúmenes de solo lectura dentro del contenedor.

## Flujo de datos

1. Los CSV originales se conservan en `data/raw/instacart/`.
2. Los scripts de `ml/training/` analizan y filtran las compras.
3. Se genera `training_baskets_v1_compact.parquet`.
4. El script de entrenamiento calcula coocurrencias y scores item-item.
5. El resultado se escribe en `item_item_recommender_v1.parquet`.
6. `LocalItemItemRecommender` carga el artifact y el catálogo de productos.
7. FastAPI recibe un carrito y devuelve productos recomendados enriquecidos.
8. Cada request exitoso a `/recommend` intenta guardar un log en PostgreSQL.
9. `/metrics/summary` agrega esos logs para presentar métricas operativas.

El fallo al insertar un log no interrumpe una recomendación ya calculada. En
cambio, si PostgreSQL no está disponible al consultar métricas,
`/metrics/summary` responde con HTTP 503.

## API actual

| Método | Ruta | Propósito |
| --- | --- | --- |
| `GET` | `/health` | Verificar que el servicio responde. |
| `GET` | `/model-info` | Consultar tipo de modelo, modo de serving y cobertura del artifact. |
| `POST` | `/recommend` | Recomendar productos para una lista de IDs de carrito. |
| `GET` | `/metrics/summary` | Consultar volumen, latencia y cantidad promedio de recomendaciones. |

Ejemplo mínimo de entrada para `/recommend`:

```json
{
  "cart_product_ids": [24852, 21137, 47766],
  "top_k": 10
}
```

## Frontend separado

El frontend está implementado como una demo local en el repositorio/carpeta
`retail-recommender-frontend`, con React, Vite y TypeScript. Consume los
endpoints `GET /health`, `GET /model-info`, `POST /recommend` y
`GET /metrics/summary` de la API FastAPI.

En desarrollo se ejecuta en `http://127.0.0.1:5173` y espera el backend en
`http://127.0.0.1:8000`. No entrena modelos ni accede directamente a
PostgreSQL. Todavía no forma parte del Docker Compose de este repositorio.

## Estado y límites actuales

La implementación actual cubre serving local, tests de API, logging básico,
métricas agregadas, ejecución con Docker Compose y un frontend separado
implementado como demo local.

Todavía no están implementados:

- integración del frontend en Docker Compose o un despliegue productivo;
- autenticación y autorización;
- despliegue o serving en Vertex AI;
- pipeline de CI/CD;
- monitoreo avanzado, alertas o dashboards.

Estos elementos son trabajo futuro y no deben asumirse como parte del sistema
actual.
