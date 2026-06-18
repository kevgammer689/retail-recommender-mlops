# Training Data Decisions

Este documento resume las decisiones tomadas durante la preparación inicial del dataset para entrenar el sistema de recomendación de productos.

## 1. Objetivo del proyecto

El objetivo es construir un sistema de recomendación de productos tipo basket analysis para un ecommerce demo.

El flujo esperado del proyecto es:

```text
dataset transaccional
→ ingeniería de datos ligera
→ entrenamiento local del modelo
→ artifact del recomendador
→ serving en Vertex AI
→ consumo desde FastAPI
→ visualización en frontend
→ logging y feedback
```

El proyecto busca demostrar un ciclo MLOps end-to-end, no solo entrenar un modelo en un notebook.

## 2. Dataset utilizado

Se utiliza el dataset Instacart Market Basket Analysis.

Archivos principales cargados:

```text
orders.csv
order_products__prior.csv
order_products__train.csv
products.csv
aisles.csv
departments.csv
```

El archivo más grande es:

```text
order_products__prior.csv
```

Con:

```text
32,434,489 filas
3,214,874 órdenes históricas
49,677 productos
```

## 3. Uso de PostgreSQL

Inicialmente se cargaron los CSV crudos a PostgreSQL.

Tablas creadas:

```text
departments
aisles
products
orders
order_products_prior
order_products_train
```

La razón para usar PostgreSQL fue:

```text
- evitar que el proyecto dependa directamente de CSV como capa operativa;
- simular una fuente estructurada de datos;
- permitir consultas desde FastAPI en etapas posteriores;
- registrar eventos, feedback y metadata del modelo;
- acercar el proyecto a una arquitectura profesional.
```

PostgreSQL se mantiene como capa de persistencia y backend, no necesariamente como motor principal de procesamiento analítico pesado.

## 4. Validación de ingesta

Después de cargar los CSV a PostgreSQL, se creó el script:

```text
ml/training/validate_instacart_raw_tables.py
```

Validaciones realizadas:

```text
- conteo de filas por tabla;
- verificación de tablas no vacías;
- distribución de orders por eval_set;
- productos sin aisle o department válido;
- order_products sin order asociado.
```

Resultado observado:

```text
departments: 21
aisles: 134
products: 49,688
orders: 3,421,083
order_products_prior: 32,434,489
order_products_train: 1,384,617
```

No se encontraron inconsistencias referenciales en las validaciones ejecutadas.

## 5. Uso inicial de tabla sample

Se creó una tabla de prueba rápida:

```text
training_baskets_sample
```

Con:

```text
100,000 órdenes únicas
1,007,672 filas
```

Esta tabla no se definió como dataset final de entrenamiento.

Su propósito fue:

```text
- validar el flujo SQL;
- probar joins;
- tener una muestra rápida para debugging;
- evitar usar el dataset completo en pruebas iniciales.
```

## 6. Cambio de enfoque: SQL vs Polars

Se intentó crear una tabla filtrada grande en PostgreSQL.

La operación resultó pesada porque implicaba:

```text
- GROUP BY sobre 32M de filas;
- joins con varias tablas;
- creación de tabla derivada;
- creación de índices;
- validaciones posteriores costosas.
```

Aunque PostgreSQL puede manejar estos volúmenes, en Docker Desktop sobre Windows y con recursos locales limitados el flujo se vuelve lento para iterar.

Por esa razón se decidió usar:

```text
PostgreSQL → persistencia, backend y trazabilidad
Polars     → EDA, filtrado y construcción de datasets analíticos
Parquet    → formato intermedio para entrenamiento local
```

## 7. Por qué Polars

Polars se incorporó porque permite trabajar de forma similar a pandas, pero con mejor rendimiento en datasets grandes.

Ventajas para este proyecto:

```text
- lectura lazy de CSV grandes con scan_csv;
- procesamiento optimizado;
- group_by eficiente;
- escritura a parquet;
- sintaxis cómoda para EDA;
- menor presión de memoria que pandas en varios escenarios.
```

La decisión no elimina PostgreSQL. Solo separa responsabilidades.

## 8. EDA inicial con Polars

Se creó el script:

```text
ml/training/analyze_instacart_distributions.py
```

Su objetivo fue entender las distribuciones antes de definir filtros.

Resultados principales:

```text
order_products_prior rows: 32,434,489
orders prior: 3,214,874
products: 49,677
```

Distribución de tamaño de canasta:

```text
min_basket_size: 1
p25: 5
p50: 8
p75: 14
p95: 25
p99: 35
max_basket_size: 145
mean_basket_size: 10.09
```

Interpretación:

```text
- las canastas de tamaño 1 no sirven para co-ocurrencia producto-producto;
- el p95 está en 25 productos;
- el p99 está en 35 productos;
- canastas muy grandes generan demasiadas combinaciones y pueden introducir ruido.
```

Distribución de frecuencia de productos:

```text
min_purchase_count: 1
p25: 17
p50: 60
p95: 2,286
p99: 9,929
max_purchase_count: 472,565
mean_purchase_count: 652.91
```

Cobertura por umbral de frecuencia:

```text
Products >= 5 purchases: 47,618
Products >= 10 purchases: 42,512
Products >= 25 purchases: 33,696
Products >= 50 purchases: 26,686
Products >= 100 purchases: 20,067
Products >= 250 purchases: 12,686
Products >= 500 purchases: 8,290
Products >= 1000 purchases: 5,058
```

Interpretación:

```text
- productos muy raros tienen poca evidencia estadística;
- incluir todos los productos aumenta dimensionalidad y ruido;
- filtrar por frecuencia ayuda a crear una primera versión más estable.
```

## 9. Datasets procesados evaluados

### 9.1 Dataset filtrado amplio

Archivo creado:

```text
data/processed/training_baskets_filtered.parquet
```

Criterios:

```text
basket_size entre 2 y 30
product_purchase_count >= 25
```

Resultado:

```text
rows: 29,679,809
orders: 2,992,335
products: 33,696
min_basket_size: 2
max_basket_size: 30
min_product_purchase_count: 25
```

Carga estimada para pares producto-producto:

```text
estimated_total_pairs: 193,133,657
mean_pairs_per_order: 64.54
max_pairs_per_order: 435
```

Decisión:

```text
Este dataset conserva buena cobertura, pero sigue siendo pesado para entrenar una primera versión local del recomendador item-item.
```

### 9.2 Dataset v1

Archivo creado:

```text
data/processed/training_baskets_v1.parquet
```

Criterios:

```text
basket_size entre 2 y 25
product_purchase_count >= 100
```

Resultado:

```text
rows: 26,938,641
orders: 2,913,891
products: 20,067
min_basket_size: 2
max_basket_size: 25
min_product_purchase_count: 100
```

Carga estimada:

```text
estimated_total_pairs: 164,506,395
mean_pairs_per_order: 56.46
max_pairs_per_order: 300
```

Decisión:

```text
Aunque redujo el volumen, sigue siendo pesado para una primera iteración local.
```

### 9.3 Dataset v1 light

Archivo creado:

```text
data/processed/training_baskets_v1_light.parquet
```

Criterios:

```text
basket_size entre 2 y 20
product_purchase_count >= 500
```

Resultado:

```text
rows: 21,343,946
orders: 2,743,352
products: 8,290
min_basket_size: 2
max_basket_size: 20
min_product_purchase_count: 500
```

Carga estimada:

```text
estimated_total_pairs: 124,710,766
mean_pairs_per_order: 45.46
max_pairs_per_order: 190
```

Decisión:

```text
Mejor que la versión anterior, pero todavía puede ser pesado para el entrenamiento local inicial.
```

### 9.4 Dataset v1 compact

Archivo creado:

```text
data/processed/training_baskets_v1_compact.parquet
```

Criterios:

```text
basket_size entre 2 y 15
product_purchase_count >= 1000
```

Resultado:

```text
rows: 15,004,135
orders: 2,411,003
products: 5,058
min_basket_size: 2
max_basket_size: 15
min_product_purchase_count: 1000
```

Carga estimada:

```text
estimated_total_pairs: 77,092,308
mean_pairs_per_order: 31.98
max_pairs_per_order: 105
```

Decisión:

```text
Este dataset se selecciona como candidato inicial para entrenar el primer modelo item-item local.
```

## 10. Justificación del dataset v1 compact

El objetivo de esta versión no es maximizar performance final.

El objetivo es crear una versión manejable para:

```text
- entrenar un primer recomendador;
- generar un artifact;
- validar la lógica de inferencia;
- construir FastAPI;
- conectar el frontend;
- desplegar posteriormente en Vertex AI.
```

Los filtros aplicados tienen justificación empírica:

```text
basket_size <= 15:
- reduce la explosión combinatoria de pares producto-producto;
- mantiene canastas de tamaño frecuente;
- baja el máximo de pares por orden a 105.

product_purchase_count >= 1000:
- reduce el catálogo a productos con suficiente evidencia;
- evita productos con asociaciones poco confiables;
- reduce el tamaño del artifact final;
- facilita servir recomendaciones en API.
```

## 11. Consideraciones metodológicas

Estos filtros no son definitivos.

En un proyecto productivo real deberían ajustarse con base en:

```text
- métricas offline: precision@k, recall@k, hit rate, MAP@k;
- cobertura de productos;
- cobertura de usuarios;
- diversidad de recomendaciones;
- estabilidad de asociaciones;
- performance de inferencia;
- objetivos de negocio;
- capacidad de cómputo disponible.
```

Para este proyecto de aprendizaje, se prioriza avanzar de forma controlada hacia un flujo MLOps funcional.

## 12. Próximo paso

Entrenar un recomendador item-item por co-ocurrencia usando:

```text
data/processed/training_baskets_v1_compact.parquet
```

Artifact esperado:

```text
artifacts/models/item_item_recommender_v1.parquet
```

Columnas esperadas:

```text
source_product_id
recommended_product_id
cooccurrence_count
score
rank
```

La API no necesita todos los pares producto-producto. Solo necesita los top-N productos recomendados por cada producto base.

Por eso, durante entrenamiento se debe:

```text
1. construir pares producto-producto;
2. contar co-ocurrencias;
3. calcular score;
4. ordenar recomendaciones por producto;
5. guardar solo top-N por source_product_id.
```