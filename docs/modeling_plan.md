# Plan de modelado actual

## Dataset

El proyecto utiliza **Instacart Market Basket Analysis**, que contiene órdenes,
productos y señales de recompra. Los datos originales se conservan en
`data/raw/instacart/`.

Para el entrenamiento productivo actual se prepara:

```text
data/processed/training_baskets_v1_compact.parquet
```

El script `ml/training/build_training_baskets_v1_compact_parquet.py` aplica
estos filtros:

- productos con al menos 1000 compras;
- canastas con un tamaño entre 2 y 15 productos;
- conservación de campos necesarios para analizar órdenes y pares de productos.

El objetivo del filtrado es reducir el número de combinaciones producto-producto
y mantener un entrenamiento local manejable. No busca maximizar todavía la
calidad predictiva final.

## Modelo principal

El modelo servido actualmente es un recomendador **item-item por
coocurrencia**. Es un algoritmo estadístico de recomendación, no un modelo
supervisado complejo.

Para cada producto base `A` y candidato `B`, el score conceptual es:

```text
score(A -> B) = cooccurrence(A, B) / frequency(A)
```

Donde:

- `cooccurrence(A, B)` es la cantidad de órdenes en las que ambos aparecen;
- `frequency(A)` es la frecuencia de compra del producto base;
- un score mayor indica que `B` aparece con mayor frecuencia cuando aparece `A`.

El entrenamiento genera pares dirigidos. Para una canasta `[A, B, C]` se
consideran `A -> B`, `A -> C`, `B -> A`, `B -> C`, `C -> A` y `C -> B`.
Después se ordenan los candidatos por score y coocurrencia, y se conserva un
máximo de 20 recomendaciones por producto base.

## Justificación

Este enfoque se eligió porque:

- es simple de implementar y validar;
- es interpretable;
- permite revisar directamente coocurrencias, scores y rankings;
- es suficiente para practicar el ciclo MLOps completo;
- genera un artifact compacto y fácil de servir sin entrenar en cada request.

## Artifact

El resultado del entrenamiento se guarda en:

```text
artifacts/models/item_item_recommender_v1.parquet
```

Columnas principales:

| Columna | Descripción |
| --- | --- |
| `source_product_id` | Producto del carrito que origina la recomendación. |
| `recommended_product_id` | Producto candidato recomendado. |
| `cooccurrence_count` | Cantidad de coocurrencias observadas. |
| `score` | Coocurrencia normalizada por la frecuencia del producto base. |
| `rank` | Posición del candidato dentro del producto base. |

Durante serving, las recomendaciones de todos los productos del carrito se
combinan, se excluyen productos ya presentes, se suman score y coocurrencias, y
se calcula cuántos productos del carrito apoyaron cada candidato.

## Exploración supervisada

Existe una exploración de predicción de recompra en
`notebooks/reorder_prediction_model_walkthrough.ipynb`. Ese trabajo estudia un
problema supervisado diferente, pero no es el modelo productivo consumido por
FastAPI.

## Prioridad actual

El foco actual no es tuning avanzado ni comparación exhaustiva de modelos. La
prioridad es consolidar:

- serving reproducible;
- contratos API y tests;
- ejecución local con Docker;
- logging operativo;
- métricas básicas;
- documentación y operación mantenible.

Las mejoras futuras del modelo deberán incluir una estrategia explícita de
evaluación offline y comparación contra este baseline antes de reemplazar el
artifact actual.
