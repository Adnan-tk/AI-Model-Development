from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.db import transaction

from .models import Product, Cart, CartItem, Wishlist, WishlistItem, Order, OrderItem
from .serializers import (
    ProductSerializer, CartSerializer, CartItemSerializer,
    WishlistSerializer, WishlistItemSerializer,
    OrderSerializer
)
from .permissions import IsAdminOrReadOnly


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.filter(is_active=True).order_by('-created_at')
    serializer_class = ProductSerializer
    permission_classes = [IsAdminOrReadOnly]
    lookup_field = 'slug'


class CartViewSet(viewsets.ModelViewSet):
    serializer_class = CartSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Cart.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'], url_path='add-item')
    def add_item(self, request, pk=None):
        cart = self.get_object()
        serializer = CartItemSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        product = serializer.validated_data['product']
        quantity = serializer.validated_data.get('quantity', 1)
        item, _ = CartItem.objects.get_or_create(cart=cart, product=product)
        item.quantity = item.quantity + quantity
        item.save()
        return Response(CartSerializer(cart).data)

    @action(detail=True, methods=['post'], url_path='remove-item')
    def remove_item(self, request, pk=None):
        cart = self.get_object()
        serializer = CartItemSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        product = serializer.validated_data['product']
        CartItem.objects.filter(cart=cart, product=product).delete()
        return Response(CartSerializer(cart).data)

    @action(detail=True, methods=['post'], url_path='update-item')
    def update_item(self, request, pk=None):
        cart = self.get_object()
        serializer = CartItemSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        product = serializer.validated_data['product']
        quantity = serializer.validated_data['quantity']
        item = get_object_or_404(CartItem, cart=cart, product=product)
        item.quantity = quantity
        item.save()
        return Response(CartSerializer(cart).data)


class WishlistViewSet(viewsets.ModelViewSet):
    serializer_class = WishlistSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Wishlist.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'], url_path='add-item')
    def add_item(self, request, pk=None):
        wishlist = self.get_object()
        serializer = WishlistItemSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        product = serializer.validated_data['product']
        WishlistItem.objects.get_or_create(wishlist=wishlist, product=product)
        return Response(WishlistSerializer(wishlist).data)

    @action(detail=True, methods=['post'], url_path='remove-item')
    def remove_item(self, request, pk=None):
        wishlist = self.get_object()
        serializer = WishlistItemSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        product = serializer.validated_data['product']
        WishlistItem.objects.filter(wishlist=wishlist, product=product).delete()
        return Response(WishlistSerializer(wishlist).data)


class OrderViewSet(viewsets.ModelViewSet):
    serializer_class = OrderSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Order.objects.filter(user=self.request.user).order_by('-created_at')

    @action(detail=False, methods=['post'], url_path='checkout')
    @transaction.atomic
    def checkout(self, request):
        cart_id = request.data.get('cart_id')
        shipping_address = request.data.get('shipping_address', '')
        cart = get_object_or_404(Cart, id=cart_id, user=request.user)
        order = Order.objects.create(user=request.user, shipping_address=shipping_address)
        total = 0
        for item in cart.items.select_related('product'):
            if item.product.stock < item.quantity:
                transaction.set_rollback(True)
                return Response({"detail": f"Insufficient stock for {item.product.name}"}, status=status.HTTP_400_BAD_REQUEST)
            item.product.stock -= item.quantity
            item.product.save()
            OrderItem.objects.create(
                order=order,
                product=item.product,
                quantity=item.quantity,
                price_at_purchase=item.product.price,
            )
            total += item.product.price * item.quantity
        order.total_amount = total
        order.status = 'paid'
        order.save()
        cart.items.all().delete()
        return Response(OrderSerializer(order).data, status=status.HTTP_201_CREATED)
