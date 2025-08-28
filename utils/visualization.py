import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import streamlit as st


class VisualizationUtils:
    def __init__(self):
        pass

    def create_decision_tree(self):
        """Create a simplified decision tree visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw the tree structure
        self._draw_node(ax, 0.5, 0.9, "Input\nText", "lightblue")
        self._draw_node(ax, 0.3, 0.7, "Tokenize", "lightgreen")
        self._draw_node(ax, 0.7, 0.7, "Clean Text", "lightgreen")
        self._draw_node(ax, 0.2, 0.5, "Embed", "lightyellow")
        self._draw_node(ax, 0.4, 0.5, "Positional\nEncoding", "lightyellow")
        self._draw_node(ax, 0.6, 0.5, "Attention", "lightyellow")
        self._draw_node(ax, 0.8, 0.5, "Feed\nForward", "lightyellow")
        self._draw_node(ax, 0.5, 0.3, "Output\nProjection", "lightcoral")
        self._draw_node(ax, 0.5, 0.1, "Generated\nResponse", "lightpink")

        # Draw connections
        self._draw_connection(ax, 0.5, 0.9, 0.3, 0.7)
        self._draw_connection(ax, 0.5, 0.9, 0.7, 0.7)
        self._draw_connection(ax, 0.3, 0.7, 0.2, 0.5)
        self._draw_connection(ax, 0.3, 0.7, 0.4, 0.5)
        self._draw_connection(ax, 0.7, 0.7, 0.6, 0.5)
        self._draw_connection(ax, 0.7, 0.7, 0.8, 0.5)
        self._draw_connection(ax, 0.2, 0.5, 0.5, 0.3)
        self._draw_connection(ax, 0.4, 0.5, 0.5, 0.3)
        self._draw_connection(ax, 0.6, 0.5, 0.5, 0.3)
        self._draw_connection(ax, 0.8, 0.5, 0.5, 0.3)
        self._draw_connection(ax, 0.5, 0.3, 0.5, 0.1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Simplified Decision Process of the Chatbot', fontsize=16)

        return fig

    def _draw_node(self, ax, x, y, text, color):
        """Draw a node in the decision tree"""
        circle = patches.Circle((x, y), 0.06, facecolor=color, edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)

    def _draw_connection(self, ax, x1, y1, x2, y2):
        """Draw a connection between nodes"""
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1)

    def plot_training_history(self, train_losses, val_losses):
        """Plot training history"""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)

        return fig

    def plot_attention_weights(self, attention_weights):
        """Plot attention weights (simplified)"""
        fig, ax = plt.subplots(figsize=(10, 8))

        if attention_weights is not None:
            im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            ax.set_title('Attention Weights')
        else:
            ax.text(0.5, 0.5, 'Attention weights not available\nin visualization mode',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        return fig