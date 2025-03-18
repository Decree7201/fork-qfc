terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-02-42e57b77448f-terraform-state"
    prefix = "prod"
  }
}
