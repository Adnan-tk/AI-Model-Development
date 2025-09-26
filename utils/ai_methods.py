import streamlit as st


class AIMethodologyExplainer:
    def __init__(self):
        self.methods = {
            "ai": {
                "title": "Artificial Intelligence (AI)",
                "description": "The broad field of creating systems that can perform tasks requiring human intelligence.",
                "key_concepts": [
                    "Problem solving",
                    "Knowledge representation",
                    "Planning",
                    "Learning",
                    "Natural language processing",
                    "Perception",
                    "Robotics"
                ],
                "examples": [
                    "Expert systems",
                    "Game playing AI (Chess, Go)",
                    "Speech recognition",
                    "Computer vision"
                ],
                "pros": ["Broad applicability", "Can solve complex problems"],
                "cons": ["Can be computationally expensive", "May lack transparency"]
            },
            "ml": {
                "title": "Machine Learning (ML)",
                "description": "A subset of AI focused on developing algorithms that can learn from and make predictions on data.",
                "key_concepts": [
                    "Supervised learning",
                    "Unsupervised learning",
                    "Reinforcement learning",
                    "Feature engineering",
                    "Model evaluation",
                    "Cross-validation"
                ],
                "examples": [
                    "Spam detection",
                    "Recommendation systems",
                    "Predictive analytics",
                    "Fraud detection"
                ],
                "algorithms": [
                    "Linear Regression",
                    "Decision Trees",
                    "Random Forests",
                    "Support Vector Machines",
                    "K-Means Clustering"
                ],
                "pros": ["Adapts to new data", "Can discover patterns in large datasets"],
                "cons": ["Requires large amounts of data", "Can be prone to bias"]
            },
            "dl": {
                "title": "Deep Learning (DL)",
                "description": "A subset of machine learning using neural networks with multiple layers to learn representations of data.",
                "key_concepts": [
                    "Neural networks",
                    "Backpropagation",
                    "Convolutional neural networks (CNNs)",
                    "Recurrent neural networks (RNNs)",
                    "Transformers",
                    "Transfer learning"
                ],
                "examples": [
                    "Image recognition",
                    "Speech recognition",
                    "Natural language processing",
                    "Autonomous vehicles"
                ],
                "architectures": [
                    "Multi-layer Perceptrons (MLPs)",
                    "Convolutional Neural Networks (CNNs)",
                    "Recurrent Neural Networks (RNNs)",
                    "Transformers",
                    "Autoencoders",
                    "Generative Adversarial Networks (GANs)"
                ],
                "pros": ["Can handle very complex patterns", "Automatic feature extraction"],
                "cons": ["Requires massive amounts of data", "Computationally intensive"]
            },
            "nn": {
                "title": "Neural Networks (NN)",
                "description": "Computational models inspired by the human brain, consisting of interconnected nodes (neurons).",
                "key_concepts": [
                    "Neurons and layers",
                    "Activation functions",
                    "Weights and biases",
                    "Loss functions",
                    "Optimization algorithms",
                    "Gradient descent"
                ],
                "examples": [
                    "Pattern recognition",
                    "Function approximation",
                    "Time series prediction",
                    "Classification tasks"
                ],
                "types": [
                    "Feedforward Neural Networks",
                    "Radial Basis Function Networks",
                    "Kohonen Self-Organizing Maps",
                    "Modular Neural Networks"
                ],
                "pros": ["Can approximate any function", "Handle non-linear relationships well"],
                "cons": ["Black box nature", "Can overfit if not regularized"]
            }
        }

    def show_methodology_info(self, method_key):
        """Display information about a specific AI methodology"""
        if method_key not in self.methods:
            st.error(f"Unknown methodology: {method_key}")
            return

        method = self.methods[method_key]

        st.header(method["title"])
        st.write(method["description"])

        st.subheader("Key Concepts")
        for concept in method["key_concepts"]:
            st.write(f"- {concept}")

        if "algorithms" in method:
            st.subheader("Common Algorithms")
            for algorithm in method["algorithms"]:
                st.write(f"- {algorithm}")

        if "architectures" in method:
            st.subheader("Common Architectures")
            for architecture in method["architectures"]:
                st.write(f"- {architecture}")

        if "types" in method:
            st.subheader("Types")
            for type_name in method["types"]:
                st.write(f"- {type_name}")

        st.subheader("Examples")
        for example in method["examples"]:
            st.write(f"- {example}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Strengths")
            for pro in method["pros"]:
                st.write(f"✅ {pro}")

        with col2:
            st.subheader("Limitations")
            for con in method["cons"]:
                st.write(f"❌ {con}")

    def show_comparison(self):
        """Show comparison between different AI methodologies"""
        st.header("Comparison of AI Methodologies")

        comparison_data = []
        for key, method in self.methods.items():
            comparison_data.append({
                "Methodology": method["title"],
                "Description": method["description"],
                "Data Requirements": "High" if key in ["ml", "dl"] else "Medium",
                "Computational Needs": "High" if key in ["dl"] else "Medium" if key in ["ml", "nn"] else "Low",
                "Interpretability": "Low" if key in ["dl", "nn"] else "Medium" if key in ["ml"] else "High"
            })

        st.table(comparison_data)

        st.subheader("When to Use Which Approach")
        st.write("""
        - **Traditional AI**: Good for rule-based systems with clear logic
        - **Machine Learning**: Best when you have structured data and clear patterns
        - **Neural Networks**: Suitable for complex pattern recognition tasks
        - **Deep Learning**: Ideal for unstructured data (images, text, audio) and very complex patterns
        """)

    def show_methodology_selector(self):
        """Show a selector for different AI methodologies"""
        st.sidebar.header("AI Methodology Explorer")

        method_option = st.sidebar.selectbox(
            "Select a methodology to learn about:",
            ["Overview"] + [self.methods[key]["title"] for key in self.methods]
        )

        if method_option == "Overview":
            self.show_comparison()
        else:
            # Find the key for the selected method
            for key, method in self.methods.items():
                if method["title"] == method_option:
                    self.show_methodology_info(key)
                    break

