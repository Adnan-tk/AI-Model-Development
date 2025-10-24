from rest_framework import routers
from django.urls import path, include

from .views import ProductViewSet, CartViewSet, WishlistViewSet, OrderViewSet

router = routers.DefaultRouter()
router.register(r'products', ProductViewSet, basename='product')
router.register(r'carts', CartViewSet, basename='cart')
router.register(r'wishlists', WishlistViewSet, basename='wishlist')
router.register(r'orders', OrderViewSet, basename='order')

urlpatterns = [
    path('', include(router.urls)),
]
