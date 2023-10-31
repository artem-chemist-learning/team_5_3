# Databricks notebook source
# MAGIC %md
# MAGIC # Setting up your Team's Cloud Storage on Azure 
# MAGIC
# MAGIC Each team will need to create a blob storage area  on Azure. This will be created by one team member known as the Storage Team Lead (possibly with an another team member as an observer). Once the blob storage is created the Storage Lead Person will give access to all other team members via shared secrets (read on to learn more). Then all team members (and only team members) will be have  access to the team storage bucket. Read on to learn how to do this. 
# MAGIC
# MAGIC ## Create storage bucket (performed by Storage Lead on your project team)
# MAGIC The Storage Lead Person in your team will need to perform the following steps to create storage bucket for your team:
# MAGIC
# MAGIC * Download Databricks CLI to your laptop
# MAGIC * Create Azure Blob Storage
# MAGIC * Generate access credentials via **SAS Token**
# MAGIC * Share Blob storage access Credentials via Databricks Secrets
# MAGIC * Set up a code cell that can be pasted into any notebook that is used by the project team thereby giving them access to the team's blob storage
# MAGIC
# MAGIC ## Read/Write to cloud storage from DataBricks cluster (can be tested by any team member)
# MAGIC Now that a blob store (aka container has been created), any member of the team can read and write from the team's blob storage (aka container in Azure jargon). 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Create storage bucket (performed by one person on your team)
# MAGIC ## Download Databricks CLI to your laptop
# MAGIC
# MAGIC **Note:** All Databricks CLI commands should be run on your `local computer`, not on the cluster.
# MAGIC
# MAGIC *  On your LOCAL LAPTOP, please install the Databricks CLI by running this command:
# MAGIC    * `python3 -m pip install databricks-cli`
# MAGIC * To access information through Databricks CLI, you have to authenticate. For authenticating and accessing the Databricks REST APIs, you have to use a personal access token. 
# MAGIC   * To generate the access token, click on the user profile icon in the top right corner of the Databricks Workspace and select user settings.
# MAGIC       * Go to the extreme top right corner of this notebook UI and make sure the NAVIGATATION BAR (which is a separate bar about the notebook menu bar) and click on the dropdown menu associated with you email **...@berkeley.edu**, then click on **User Settings**, 
# MAGIC   * Enter the name of the comment and lifetime (total validity days of the token). 
# MAGIC   * Click on generate.
# MAGIC   * Now, the Personal Access is generated; copy the generated token. 
# MAGIC     * NOTE: once you generate a token you will only have one chance to copy the token to a safe place.
# MAGIC
# MAGIC * In the command prompt, type `databricks configure –token` and press enter.
# MAGIC   *  When prompted to enter the Databricks Host URL, provide your Databricks Host Link which is `https://adb-4248444930383559.19.azuredatabricks.net`.
# MAGIC   * Then, you will be asked to enter the token. Enter your generated TOKEN and authenticate.
# MAGIC
# MAGIC * Now, you are successfully authenticated and all set for creating Secret Scopes and Secrets using CLI (see below). Secret Scopes and Secrets help to avoid sharing passwords and access keys in your notebooks. 
# MAGIC * NOTE: you can also see this TOKEN via the command line by typing the following command on your Terminal window.
# MAGIC   `Jamess-MacBook-Pro-10:~ jshanahan$      cat ~/.databrickscfg`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Azure Blob Storage and generate access priviledges
# MAGIC
# MAGIC **Special Note:** Creating a Storage account, only needs to be performed by **one** member of the team. This person then creates a blob storage area (known as a container) and shares access credentials with the rest of the team via a Secrets ACL. Please be responsible.
# MAGIC
# MAGIC ### Create Storage Account
# MAGIC 1. Navigate to https://portal.azure.com
# MAGIC 2. Login using Calnet credentials *myuser@berkeley.edu*
# MAGIC 3. Click on the top right corner on the User Icon.
# MAGIC 4. Click on Switch directory. Make sure you switch to **UC Berkeley berkeley.onmicrosoft.com**, this would be your personal space.
# MAGIC 5. Click on the Hamburger Menu Icon on the top left corner, navigate to **Storage accounts**.
# MAGIC 6. Choose the option **Azure for Students** to take advantage of $100 in credits. Provide you *berkeley.edu* email and follow the prompts.
# MAGIC 7. Once the subscription is in place, navigate back to Storage accounts, refresh if needed. Hit the button **+ Create** in the top menu.
# MAGIC   - Choose **Azure for Students** as Subscription (think billing account).
# MAGIC   - Create a new Resource group. Name is irrelevant here.
# MAGIC   - Choose a **Storage account name**, you will need this in the *Init Script* below. (e.g., jshanahan). This a master directory within which we have blob storages, aka containers on Azure.
# MAGIC   - Go with the defaults for the rest of the form.
# MAGIC   - Hit the **Review + create** button.
# MAGIC 8. Once the **Storage account** is shown in your list:
# MAGIC   - Click on it. This will open a sub-window.
# MAGIC   - Under *Data Storage*, click on **container**.
# MAGIC   - Hit the **+ Container** in the top menu.
# MAGIC   - Choose a name for your container; to access this container you will need to generate a SAS token in the *Init Script* below.
# MAGIC   
# MAGIC **Note:** Create your Blob Storage in the US West 2 Region.
# MAGIC
# MAGIC ### Obtain Credentials via **SAS Token** or via **Access Key**
# MAGIC
# MAGIC First, you need to choose between using  a SAS token (or via Access Key) to enable access to you blob storage. Bottom line, SAS tokens would be recommended since it's a token in which you have control on permissions and TTL (Time to Live). On the other hand, an Access Key, would grant full access to the Storage Account and will generate SAS tokens in the backend when these expire.
# MAGIC
# MAGIC
# MAGIC To obtain a **SAS Token** which is the recommended way to offer access to your team mates.
# MAGIC
# MAGIC SAS Token (Shared Access Signature token)  that offers access for a restricted time period which we recommend:
# MAGIC 1. Navigate to the containers list.
# MAGIC 2. At the far right, click on the `...` for the container you just created.
# MAGIC 3. Check the boxes of the permissions you want.
# MAGIC 4. Select an expiration you are comfortable with.
# MAGIC 5. Hit the **Generate SAS token and URL** button.
# MAGIC 6. Scroll down and copy only the **Blob SAS token**.
# MAGIC
# MAGIC Please try to avoid using **Access Key**:
# MAGIC
# MAGIC To obtain the **Access Key** (unrestricted access as long as you have the token):
# MAGIC 1. Navigate back to *Storage accounts**.
# MAGIC 2. Click on the recently created account name.
# MAGIC 3. In the sub-window, under *Security + networking*, click on **Access Keys**.
# MAGIC 4. Hit the **Show keys** button.
# MAGIC 5. Copy the **Key**, you don't need the Connection string. It's irrelevant if you choose *key1* or *key2*.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Share Blob storage access credentials securely via Databricks Secret  
# MAGIC
# MAGIC Now, you are successfully authenticated next we will create Secret Scopes and Secrets using the CLI to avoid sharing passwords and access keys in your notebooks.
# MAGIC
# MAGIC #### Some background on scopes and secrets [you can skip this subsection if you are low on time]
# MAGIC Since security is the primary concern when working with Cloud services, instead of storing passwords or access keys in Notebook or Code in plaintext, Databricks/Azure provides two types of Secret Scopes to store and retrieve all the secrets when and where they are needed.  In Databricks, every Workspace has Secret Scopes within which one or more Secrets are present to access third-party data, integrate with applications, or fetch information. Users can also create multiple Secret Scopes within the workspace according to the demand of the application.  
# MAGIC
# MAGIC The two types of Databricks Secret Scopes are:
# MAGIC * 1) Azure Key Vault-Backed Scope [not applicable here; see Azure documentation for more details]
# MAGIC
# MAGIC * 2) Databricks-Backed Scope
# MAGIC In this method, the Secret Scopes are managed with an internally encrypted database owned by the Databricks platform. Users can create a Databricks-backed Secret Scope using the Databricks CLI version 0.7.1 and above.
# MAGIC
# MAGIC #### Permission Levels of Secret Scopes
# MAGIC There are three levels of permissions that you can assign while creating each Secret Ccope. They are:
# MAGIC
# MAGIC * Manage: This permission is used to manage everything about the Secret Scopes and ACLS (Access Control List). By using ACLs, users can configure fine-grained permissions to different people and groups for accessing different Scopes and Secrets.
# MAGIC * Write: This allows you to read, write, and manage the keys of the particular Secret Scope.
# MAGIC * Read: This allows you to read the secret scope and list all the secrets available inside it.
# MAGIC
# MAGIC
# MAGIC ###  Creating Secret Scopes and Secrets using Databricks CLI (to avoid sharing passwords and access keys in your notebooks)
# MAGIC
# MAGIC **Special Note:** Only the member that created the Storage account should perform this step.
# MAGIC
# MAGIC 1. On your laptop via the CLI, create a **SCOPE**:
# MAGIC *  `databricks secrets create-scope --scope <choose-any-name>`
# MAGIC
# MAGIC 2. Next create Secrets inside the Secret Scope using Databricks CLI
# MAGIC You can enter the following command to create a Secret inside the Scope: On your laptop via the CLI, load the key/token:
# MAGIC * `databricks secrets put --scope <name-from-above> --key <choose-any-name> --string-value '<paste-key-SAS-token-here>'`
# MAGIC
# MAGIC NOTE --principal should be  CLUSTER Name 
# MAGIC
# MAGIC 3.  On your laptop via the CLI, add a `principal` to the Secret Scope ACL to share token with your teammates. This is done at the team cluster level, so you will need the name of your Databricks cluster.  **Careful:** make sure you type the right cluster name.
# MAGIC * `databricks secrets put-acl --scope <name-from-above> --principal "Data Bricks CLUSTER-Name"  --permission READ`
# MAGIC
# MAGIC Putting all three steps together, it might look like this for a sample project team who is running on a Databricks cluster called `team 1-1`:
# MAGIC ```bash
# MAGIC databricks secrets create-scope --scope team_1_1_scope   #made a scope of jgs_instructors;  
# MAGIC databricks secrets put --scope team_1_1_scope --key team_1_1_key \
# MAGIC         --string-value 'sp=racwdli&st=2022-11-19T21:43:t..........' #SAS Container token that you copied from Azure
# MAGIC databricks secrets put-acl --scope team_1_1_scope --principal "team 1-1" --permission READ  #assume my DataBricks cluster name is team 1-1
# MAGIC
# MAGIC ```
# MAGIC **Note:** This has been tested only on Mac/Linux. It might be different in Windows.
# MAGIC   * For Windows: to load the key/ SAS token, replace the single quote `''` with double quote `""`.
# MAGIC   `databricks secrets put --scope <name-from-above> --key <choose-any-name> --string-value "<paste-key-SAS-token-here>"`
# MAGIC
# MAGIC Then each team members could run the following, there by saving a small Spark dataframe to the team's blob storage.  Then any team member can see the saved data on the team blob storage `test` via https://portal.azure.com:
# MAGIC
# MAGIC ```python
# MAGIC secret_scope = "team_1_1_scope"
# MAGIC secret_key   = "team_1_1_key"    
# MAGIC spark.conf.set(
# MAGIC   f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
# MAGIC   dbutils.secrets.get(scope = secret_scope, key = secret_key)
# MAGIC )
# MAGIC blob_container  = "my_container_name"       # The name of your container created in https://portal.azure.com
# MAGIC storage_account = "my_storage_account_name" # The name of your Storage account created in https://portal.azure.com
# MAGIC team_blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
# MAGIC
# MAGIC pdf = pd.DataFrame([[1, 2, 3, "Jane"], [2, 2,2, None], [12, 12,12, "John"]], columns=["x", "y", "z", "a_string"])
# MAGIC df = spark.createDataFrame(pdf) # Create a Spark dataframe from a pandas DF
# MAGIC
# MAGIC # The following can write the dataframe to the team's Cloud Storage  
# MAGIC # Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
# MAGIC df.write.parquet(f"{team_blob_url}/test")
# MAGIC
# MAGIC # see what's in the parquet folder 
# MAGIC display(dbutils.fs.ls(f"{team_blob_url}/test"))
# MAGIC ```
# MAGIC

# COMMAND ----------

secret_scope = "team53scope"
secret_key   = "team53secret"    
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)
blob_container  = "team53container"       # The name of your container created in https://portal.azure.com
storage_account = "w261team53" # The name of your Storage account created in https://portal.azure.com
team_blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

pdf = pd.DataFrame([[1, 2, 3, "Jane"], [2, 2,2, None], [12, 12,12, "John"]], columns=["x", "y", "z", "a_string"])
df = spark.createDataFrame(pdf) # Create a Spark dataframe from a pandas DF

# The following can write the dataframe to the team's Cloud Storage  
# Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
df.write.parquet(f"{team_blob_url}/test")

# see what's in the parquet folder 
display(dbutils.fs.ls(f"{team_blob_url}/test"))

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # Read/write to blob storage (for all team members)
# MAGIC
# MAGIC ## Init Script 
# MAGIC The Storage Team Lead will need to adapt the following cell so that team members can read/write to the team's blob storage. 
# MAGIC
# MAGIC Please replace these variable values with your blob storage details and access credential information:
# MAGIC
# MAGIC ```python
# MAGIC blob_container  = “my_container_name”        # The name of your container created in https://portal.azure.com
# MAGIC storage_account = “my_storage_account_name”  # The name of your Storage account created in https://portal.azure.com
# MAGIC secret_scope    = “team_1_1_scope”           # The name of the scope created in your local computer using the Databricks CLI
# MAGIC secret_key      = “team_1_1_key”             # The name of the secret key created in your local computer using the Databricks CLI
# MAGIC ```
# MAGIC
# MAGIC This cell can then be copied to any team notebook that needs access to the team cloud storage.

# COMMAND ----------

## Place this cell in any team notebook that needs access to the team cloud storage.


# The following blob storage is accessible to team members only (read and write)
# access key is valid til TTL
# after that you will need to create a new SAS key and authenticate access again via DataBrick command line
blob_container  = "team53container"       # The name of your container created in https://portal.azure.com
storage_account = "w261team53"  # The name of your Storage account created in https://portal.azure.com
secret_scope    = "team53scope"           # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "team53secret"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket


# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)
import pandas as pd
pdf = pd.DataFrame([[1, 2, 3, "Jane"], [2, 2,2, None], [12, 12,12, "John"]], columns=["x", "y", "z", "a_string"])
df = spark.createDataFrame(pdf) # Create a Spark dataframe from a pandas DF

# The following can write the dataframe to the team's Cloud Storage  
# Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
df.write.parquet(f"{team_blob_url}/TP")



# see what's in the blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and write data!
# MAGIC A *Read Only* mount has been made available to all course clusters in this Databricks Platform. It contains data you will use for **HW5** and **Final Project**. Feel free to explore the files by running the cell below. Read them!
# MAGIC
# MAGIC

# COMMAND ----------

display(dbutils.fs.ls(f"{mids261_mount_path}/datasets_final_project_2022"))

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count, col, split, trim, lit, avg, sum

df_airlines = spark.read.parquet(f"{mids261_mount_path}/datasets_final_project/parquet_airlines_data_3m/")# Load the Jan 1st, 2015 for Weather
df_weather =  spark.read.parquet(f"{mids261_mount_path}/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-01-02T00:00:00000").cache()
display(df_weather)

# COMMAND ----------

# This command will write to your Cloud Storage if right permissions are in place. 
# Navigate back to your Storage account in https://portal.azure.com, to inspect the files.
df_weather.write.parquet(f"{team_blob_url}/TP")

# COMMAND ----------

# see what's in the parquet folder 
display(dbutils.fs.ls(f"{team_blob_url}/TP"))

# COMMAND ----------

# Load it the previous DF as a new DF
df_weather_new = spark.read.parquet(f"{team_blob_url}/TP")
display(df_weather_new)

# COMMAND ----------

print(f"Your new df_weather has {df_weather_new.count():,} rows.")
print(f'Max date: {df_weather_new.select([max("DATE")]).collect()[0]["max(DATE)"].strftime("%Y-%m-%d %H:%M:%S")}')

# COMMAND ----------

display(dbutils.fs.ls(f"{mids261_mount_path}/HW5"))

# COMMAND ----------

# MAGIC %md
# MAGIC # [DEPRECATED]
# MAGIC ### Using RDD API
# MAGIC
# MAGIC When reading/writing using the RDD API, configuration cannot happen at runtime but at cluster creation.
# MAGIC If you need the following information to be added in your Cluster as Spark Configuration when running RDD API, ping TA team. You normally do not need this set up for the final project.
# MAGIC - Storage Account name
# MAGIC - Container name
# MAGIC - Secret Scope name
# MAGIC - Secret Key name
# MAGIC
# MAGIC **Important:** Do not share the actual SAS token.
# MAGIC
# MAGIC After this is added as Spark Configuration, try the scripts provided below to test the Hadoop plug-in to connect to your Azure Blob Storage.
# MAGIC ```
# MAGIC spark.hadoop.fs.azure.sas.<container_name>.<storage_account>.blob.core.windows.net {{secrets/<scope>/<key>}}
# MAGIC ```

# COMMAND ----------

rdd = sc.textFile('/mnt/mids-w261/HW5/test_graph.txt')


parsed_rdd = rdd.map(lambda line: tuple(line.split('\t')))
parsed_rdd.take(3)

# COMMAND ----------

parsed_rdd.saveAsTextFile(f"{blob_url}/graph_test")

# COMMAND ----------


