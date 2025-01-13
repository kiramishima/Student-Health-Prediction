deploy_aws:
	cd IaC && cp sample.vars.tfvars vars.tfvars
	cd IaC && terraform init && terraform plan -var-file="vars.tfvars" && terraform apply -var-file="vars.tfvars"