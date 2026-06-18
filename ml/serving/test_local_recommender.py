from local_recommender import LocalItemItemRecommender


# Test cart with multiple products.
# 24852 = Banana
# 21137 = Organic Strawberries
# 47766 = Organic Avocado
SAMPLE_CART_PRODUCT_IDS = [24852, 21137, 47766]
TOP_K = 10


def main() -> None:
    print("Loading local recommender artifact and product catalog...")
    recommender = LocalItemItemRecommender()

    print(f"Input cart product IDs: {SAMPLE_CART_PRODUCT_IDS}")
    cart_products = recommender.get_cart_products(SAMPLE_CART_PRODUCT_IDS)

    print("\nCart products:")
    print(cart_products.to_string(index=False))

    recommendations = recommender.recommend_for_cart(
        SAMPLE_CART_PRODUCT_IDS,
        top_k=TOP_K,
    )

    print("\nRecommendations:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
