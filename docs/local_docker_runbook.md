# Runbook local con Docker

Guía operativa para levantar FastAPI y PostgreSQL en desarrollo local.

## Requisitos

- Docker Desktop iniciado.
- Repositorio disponible localmente.
- Puertos `8000` y `5432` libres.
- Los siguientes archivos presentes:

```text
artifacts/models/item_item_recommender_v1.parquet
data/raw/instacart/products.csv
```

Ejecutar los comandos desde la raíz del repositorio.

## Levantar servicios

Compatible con CMD y PowerShell:

```cmd
docker compose -f infra\docker\docker-compose.yml up -d --build
```

Compose levanta:

- `retail-recommender-postgres`;
- `retail-recommender-backend`.

El backend usa el hostname interno `postgres` para conectarse a la base de
datos. No usa `localhost` dentro de la red Docker.

## Ver contenedores

```cmd
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Ambos contenedores deben aparecer como `healthy`.

Para revisar logs:

```cmd
docker compose -f infra\docker\docker-compose.yml logs --tail=100 backend postgres
```

## Aplicar SQL de logs

Los scripts de `infra/sql/` se ejecutan automáticamente cuando PostgreSQL crea
un volumen vacío. Si el volumen ya existía antes de agregar el script, aplicar
la tabla manualmente.

En CMD:

```cmd
type infra\sql\004_create_recommendation_logs.sql | docker exec -i retail-recommender-postgres psql -U retail_user -d retail_recommender
```

En PowerShell:

```powershell
Get-Content infra\sql\004_create_recommendation_logs.sql | docker exec -i retail-recommender-postgres psql -U retail_user -d retail_recommender
```

El script usa `CREATE TABLE IF NOT EXISTS`, por lo que puede ejecutarse de forma
segura más de una vez.

## Probar endpoints

### Health

```cmd
curl http://127.0.0.1:8000/health
```

### Información del modelo

```cmd
curl http://127.0.0.1:8000/model-info
```

### Recomendación

```cmd
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"cart_product_ids\":[24852,21137,47766],\"top_k\":10}"
```

En PowerShell, `curl` puede ser un alias de `Invoke-WebRequest`. Si el comando
anterior no funciona, usar:

```powershell
$body = @{ cart_product_ids = @(24852, 21137, 47766); top_k = 10 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/recommend -Method Post -ContentType "application/json" -Body $body
```

### Resumen de métricas

```cmd
curl http://127.0.0.1:8000/metrics/summary
```

Ejecutar primero al menos una recomendación para observar métricas con datos.

## Check operativo de punta a punta

Con los servicios levantados, ejecutar desde la raíz del repositorio:

```cmd
python scripts\check_local_stack.py
```

Para indicar otra URL de la API:

```cmd
python scripts\check_local_stack.py --base-url http://127.0.0.1:8000
```

El script valida los archivos requeridos, los cuatro endpoints, la estructura
de las recomendaciones y que las métricas reporten al menos un request
registrado. Devuelve código `0` si todo está correcto y código `1` con un
mensaje `[FAIL]` si encuentra un problema. No inicia ni detiene Docker.

## Ver logs en PostgreSQL

```cmd
docker exec -it retail-recommender-postgres psql -U retail_user -d retail_recommender -c "SELECT id, request_id, created_at, top_k, recommendation_count, latency_ms FROM recommendation_logs ORDER BY id DESC LIMIT 5;"
```

Cada request exitoso a `/recommend` debe generar un registro. Si la escritura
del log falla, la API registra el error pero mantiene la respuesta de
recomendación.

## Apagar servicios

```cmd
docker compose -f infra\docker\docker-compose.yml down
```

Este comando conserva el volumen de PostgreSQL. No agregar `-v` si se desea
mantener los datos locales.

## Troubleshooting

### Docker Desktop está apagado

Síntoma: error al conectar con el engine de Docker.

Acción: iniciar Docker Desktop y esperar a que el engine esté disponible.

### Puerto 8000 ocupado

Síntoma: el backend no puede publicar el puerto.

Acción: detener el proceso que usa `8000` o cambiar temporalmente el puerto
publicado en Compose.

### Puerto 5432 ocupado

Síntoma: PostgreSQL no puede publicar su puerto.

Acción: detener la instancia local que usa `5432` o ajustar solo el puerto del
host. El backend debe seguir conectándose al servicio `postgres:5432`.

### Falta el artifact

Síntoma: el backend falla durante el arranque con `Model artifact not found`.

Acción: confirmar:

```text
artifacts/models/item_item_recommender_v1.parquet
```

### Falta `products.csv`

Síntoma: el backend falla durante el arranque con `Products file not found`.

Acción: confirmar:

```text
data/raw/instacart/products.csv
```

### `DATABASE_URL` incorrecto

Dentro de Docker el host debe ser `postgres`, no `localhost`. Revisar la
configuración efectiva:

```cmd
docker compose -f infra\docker\docker-compose.yml config
```

No publicar ni copiar credenciales reales al compartir esa salida.

### No existe `recommendation_logs`

Síntoma: `/metrics/summary` responde HTTP 503 o el backend reporta errores al
persistir logs.

Acción: aplicar `infra/sql/004_create_recommendation_logs.sql` con el comando de
CMD o PowerShell indicado arriba.

## Funcionalidad fuera del alcance actual

Este entorno todavía no incluye frontend, autenticación, Vertex AI, CI/CD ni
monitoreo avanzado con alertas o dashboards.
