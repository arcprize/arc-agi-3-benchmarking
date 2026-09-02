# Models

***

### What is Kaggle Models

[Kaggle Models](https://www.kaggle.com/models) provides a way to discover, use, and share models for machine learning and generative AI applications. Kaggle Models is a repository of pre-trained models that are deeply integrated with Kaggle's platform, making them easy to use in Kaggle Competitions and Notebooks. Like Datasets, Kaggle Models organize community activity that enrich models' usefulness: every model page will contain discussions, public notebooks, and usage statistics like downloads and upvotes that make models more useful.

#### Where do Models come from?

Kaggle Models come from a variety of sources including partners that we collaborate with on releases like Meta's Llama 2 and Alibaba's Qwen, integrations with modeling libraries like Keras, [integrations with Hugging Face Hub](https://www.kaggle.com/blog/kaggle-hugging-face-integration), and the community of millions of Kagglers sharing fine-tuned variants and other innovations.

***

### Finding Kaggle Models

You can find Kaggle Models by using the [Models landing page](https://www.kaggle.com/models). There are a number of filters and sorts plus free text search. For instances you can search by:

- Filtering by Organization, Community, or Hugging Face models
- Filtering by framework
- Filtering by the task tag you want (e.g., classification)
- Filtering by model size
- Searching by keywords in the free text search
- Sorting by number of upvotes
- Etc.

You may also want to peruse the "Models" tab on competitions to see what models are performing well or are otherwise popular for tasks relevant to your use case. Competitors commonly share which models they're using in public notebooks and in discussion write-ups. When you fork a notebook that has a model from Kaggle Models attached to it, your copy will also have the same model attached.

Finally, you can also search for models from within the notebook editor. Use the "Add Models" component in the right-hand pane of the editor to search and attach models to your notebooks. This works similarly to Datasets.

#### Understanding the model detail page

When you click on a model you will be taken to the "detail page" for that model. For example, this is the detail page for a [BERT model](https://www.kaggle.com/models/google/bert). The model detail page contains an overview tab with a Model Card (metadata and information about how the model was trained, what its acceptable use cases are, any limitations, etc.), a framework and variation explorer, and a usage dashboard. There are tabs for notebooks and discussions. If a model is useful, you can upvote it.

Beyond the overall metadata, a model detail page also organizes all variations and frameworks for a given model. For example:

- **Variations**: The same model with different numbers of parameters, e.g., small, medium, and large.
- **Frameworks**: The same model with different ML library compatibility, e.g., TensorFlow, PyTorch, etc.

You can view and use the specific framework and variation that you want by selecting it in the file explorer on the overview page beneath the Model Card. From here, you can use click "New Notebook" to attach it to a new notebook to start using the model.

### Using Kaggle Models

There's two broad ways that Kaggle Models are useful: on Kaggle and outside of Kaggle (e.g., in production applications or using non-Kaggle tools like Colab, etc.).

**On Kaggle**

Currently, Kaggle Models are very useful within the context of Competitions, specifically for use within Notebooks. Start by either forking a notebook that has a model attached (you can view the attached models on the "Input" tab of any notebook), creating a new notebook on a model, or adding a model to a new notebook from the right-hand pane of the editor.

You'll be prompted to confirm your framework and model variations(s), then simply copy and paste the starter code to load the model.

If you are downloading a Hugging Face model in your notebook, e.g., by using the Transformers library, you don't have to do anything special to use Kaggle Models. A model page will be automatically "attached" to your notebook for you.

**Outside of Kaggle**

Many developers will need to download models in code outside of Kaggle. There are a few different methods: via the [kagglehub Python library](https://github.com/Kaggle/kagglehub), via our [Kaggle CLI](https://github.com/Kaggle/kaggle-cli), or by calling the API directly.

Before providing instructions for each of these methods, it's helpful to know that you will need to know how to authenticate in order to access certain models like [Gemma](https://www.kaggle.com/models/google/gemma) that require Kaggle credentials in order to confirm that user consent to the custom license has been verified. [Obtain credentials](https://www.kaggle.com/settings) from the “Settings” page when logged-in to Kaggle and clicking on the "Generate New Token" button under the "API" section.

The examples below allow you to download the `2b` PyTorch variation for the [google/gemma](https://www.kaggle.com/models/google/gemma) model. If a model doesn't have a restricted license like Gemma, you'll be able to skip the `kagglehub.login()` steps in the examples below.

#### Method 1. Via the kagglehub Python library

See [kagglehub model download documentation](https://github.com/Kaggle/kagglehub?tab=readme-ov-file#download-model).

#### Method 2. Via the Kaggle CLI

See [Kaggle CLI model download documentation](https://github.com/Kaggle/kagglehub?tab=readme-ov-file#download-model).

#### Method 3. Calling the API directly

```
# Authenticate with credentials
export KAGGLE_USERNAME=xyz
export KAGGLE_KEY=xyz

# With Curl
curl -L -o ~/Downloads/model.tar.gz  https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/1/download -u $KAGGLE_USERNAME:$KAGGLE_KEY

# Download specific version (here version 1)
wget https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/1/download --user=$KAGGLE_USERNAME --password=$KAGGLE_KEY --auth-no-challenge
```

***

### Creating a Model

There are a few ways to publish a model on Kaggle Models including exclusively via the UI. We recommend using a combination of `kagglehub`, our Python client library, to manage artifact creation and uploads and the UI to manage documentation and collaborative features. And, if you want to use a Hugging Face model on Kaggle, you simply need to create a notebook that uses the model, e.g., in Transformers, and a model page on Kaggle will be created automatically for you.

#### Uploading using kagglehub Python client library (preferred)

See [kagglehub model upload documentation](https://github.com/Kaggle/kagglehub?tab=readme-ov-file#upload-model).

#### Uploading using the Kaggle CLI

See [Kaggle CLI model creation tutorial](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md#tutorial-create-a-model-variation).

#### Upload via the UI

1. Go to: [https://www.kaggle.com/models?new=true](https://www.kaggle.com/models?new=true) and follow the steps including setting “Creating As” to the Organization Profile you want to publish under
2. To add new Variations once your model is initially created:
   1. Scroll down to the "Model Variations" section.
   2. Click on the "New Variation" button to open the "Add/Edit" Variations modal.
   3. Select the ML framework for which you want to update weights / assets for.
   4. Click on the "Add new variation" button
   5. Select the weight / assets files to upload
   6. Enter the variation slug
      1. For example, `7b`
      2. Select a license
   7. Click on the "Create" button and wait until your instance has been fully processed.
   8. Click on "Go to model detail page".
   9. In the "Model Variations" section, you should see your variation in the drop-down.
   10. If you select it, confirm that you have all the files you were expecting under the "File Explorer" section.
   11. To upload a new version for an existing variation. Use the "New Version" button.

#### Create via Hugging Face Integration

If you have published a model on Hugging Face Hub, you can create a page for it on Kaggle easily by simply using your model in a Kaggle Notebook.

1. Navigate to your model page on Hugging Face
2. Click "Use this model" in the page header
3. Select "Kaggle" from the drop down to create a Kaggle Notebook
4. If you're not already logged-in, you will be prompted to do so
5. Optionally make changes to your notebook
6. Give your notebook a name and create a "Save Version"
7. Optionally click "Share" to make your notebook public

Once you complete these steps, a page for your model will be automatically created. Any public notebooks using the model will show up in the "Code" tab on the model page.

#### Documenting models

Documenting your model is easiest to do via the UI.

1. When viewing your model page, you will see a section at the top called “Pending Actions”.
2. Follow each of these steps to complete your model’s documentation:
   1. Add a description (model card)
   2. Add model instance descriptions including example code
   3. Add a subtitle
   4. Add tags
   5. Specify provenance and other metadata
   6. Publish a notebook (we recommend making it public after your model is made public)
3. Once your model is made public, you can also optionally generate a DOI from the “Metadata” section of your model.
4. Once you’re done, you can make your model public from the “Settings” tab on the model page.
5. You can now promote your model!
6. You’ll be automatically subscribed to email and site notifications when any discussion topics are created

#### Importing Model Versions

This tool allows you to copy model versions from one model to another. You can import versions from any public model or private models where you have collaborator access.

1. Navigate to your target model's page on Kaggle
   1. Click the "︙" button in the top right
   2. Select "Import Versions" from the dropdown menu
2. Select Source Model:
   1. Click the "Select Model" button
   2. Browse or search for the model you want to import versions from
      1. You can only select models you own or have collaborator access to
      2. The current model will be disabled to prevent self-copying
   3. Click on your chosen model to select it
3. Select Versions:
   1. Once you've selected a source model, you'll see all available versions
   2. Use the search bar to filter versions by name
   3. Use the framework chips to filter by specific frameworks (PyTorch, TensorFlow, etc.)
   4. Select versions by checking the boxes in the leftmost column
      1. You can select multiple versions at once
   5. Click "Next" to proceed to confirmation
4. Confirm and Import:
   1. Review the versions you selected
      1. Each row shows the full path of what will be copied
      2. Source path → Target path is displayed
   2. Important notes:
      1. If importing from a private model to a public model, versions will become public permanently
      2. This action cannot be undone
   3. Click "Import" to begin the copy process
5. After Import:
   1. A progress indicator will show while versions are being copied
   2. For successful imports:
      1. You'll see a success message
      2. Click "Go to Model" to view your imported versions
   3. If any versions fail to import:
      1. Error messages will explain what went wrong
      2. You can retry failed imports by clicking "Try again"
      3. Successfully copied versions will not be duplicated on retry

##### Tips for Importing

- You can swap the source model at any time using the "Swap model" button
- Use framework filters to quickly find specific versions
- The version count shows how many items you've selected
- All imported versions maintain their original framework and variation slugs

### How to name your model and variations

A handle is represented as

###### owner\_slug/model\_slug/framework/variation\_slug/version\_number

The breakdown is as follows:

- **owner\_slug:** Your organization or username.
- **model\_slug:** The name of your model family (e.g., "llama").
- **framework:** The model framework used (e.g., "pytorch").
- **variation\_slug:** Details about this specific version of your model.
- **version\_number:** A numeric identifier for tracking model changes.

#### Model vs. Variation: Uniqueness

A variation is used to add finer level details about a model. A variation should capture the intricacies and nuances of a model. They highlight specific changes or features. Examples include:

- **Model Size:** Number of parameters (e.g., 7 billion)
- **Optimization:** Quantization (e.g., int4), model distillation
- **Task:** What your model does (e.g., image generation, translation, chat)
- **Training:** Specific techniques used (e.g., instruction-tuned, prompt-tuned)
- **Architecture/Code Modifications:** Any changes from the base model
- **Dataset:** The data it was trained on (if relevant)
- **Language:** If your model is language-specific (e.g., "en" for English)
- **Hardware:** Optimized for GPU, CPU, TPU, etc.

#### Version vs. Variation: Snapshots in Time

Versions are like checkpoints. They represent a model at a specific point in training, usually with all other factors (the variation details) held constant.

#### Questions to Guide Your Variation Naming:

1. How large is your model (number of parameters)? ex: 100m, 2b, 27b, etc..
2. What task does it perform? ex: image generation, text, chat
3. What dataset was it trained on? ex: coco, imagenet
4. Did you make any changes to the code, architecture, or configuration?
5. What training techniques did you use? ex: Instruction Tuned, Prompt Tuned, etc…
6. Is it optimized for a specific language or hardware? ex: gpu, cpu, tpu
7. Did you apply any quantization or other optimizations?

#### General Guidelines for Naming Success:

- **Keep it Simple:** Use clear, concise names.
- **Be Descriptive:** Use the questions above to guide you.
- **Default to Model Name:** If unsure, use the model name as the variation too.
- **Version for Checkpoints:** Use the version number to track training progress.

#### Real-World Examples

| Handle | Variation Note |
| --- | --- |
| google/gemma-2/gguf/2.0-27b-it/1 | Version 2, 27 billion parameters, instruction tuned |
| google/gemma/tfLite/gemma-2b-it-gpu-int4/1 | 2 billion , instruction tuned, gpu, int4 quantization |
| metaresearch/llama-3/pyTorch/70b-chat | 70 billion parameters, chating |
| mistral-ai/mistral/pyTorch/7b-v0.1-hf | 7 billion parameters, version 0.1 |
| deepmind/biggan/tensorFlow1/128 | 128 x 128 image generation |

### Accessing Gated Models

A gated model on Kaggle requires users to agree to a specific agreement and potentially provide information before they can access it. This agreement can include terms of use, privacy policy links, and a form for collecting user data.

When accessing a gated model, users will be prompted to input information based on the access agreement. A banner will display the user's current access status (e.g., requiring a consent, pending, accepted, rejected). Only users with "accepted" status can proceed to use the model.
