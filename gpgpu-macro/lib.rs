use proc_macro::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use std::collections::VecDeque;
use std::str::FromStr;

#[proc_macro_attribute]
pub fn entry(_: TokenStream, item: TokenStream) -> TokenStream {
    let mut new_item = item.to_string();

    if cfg!(feature = "cpu") {
        // make sure spirv_std is imported.
        let mut import_spirv_std = "extern crate spirv_std;".to_string();
        import_spirv_std.push_str(&new_item);
        new_item = import_spirv_std;

        let body_start = new_item.chars().position(|c| c == '{').unwrap() + 1;
        new_item.insert_str(body_start, "gpgpu_backend::init();");
        TokenStream::from_str(&new_item).unwrap()
    } else {
        TokenStream::new()
    }
}

#[proc_macro_attribute]
pub fn cpu_only(_: TokenStream, item: TokenStream) -> TokenStream {
    if cfg!(feature = "cpu") {
        item
    } else {
        TokenStream::new()
    }
}

fn find_and_strip_return_type(item: TokenStream) -> (TokenStream, Option<String>) {
    // find and strip return type
    let mut return_type = None;
    let mut tokens: Vec<TokenTree> = Vec::new();
    for tt in item.clone() {
        if tokens.len() > 2
            && tokens[tokens.len() - 2].to_string() == "-"
            && tokens[tokens.len() - 1].to_string() == ">"
        {
            return_type = Some(tt.to_string());
            tokens.pop();
            tokens.pop();
        } else {
            tokens.push(tt);
        }
    }
    (tokens.into_iter().collect(), return_type)
}

#[allow(dead_code)]
fn gather_parameters(item: &TokenStream) -> Vec<(String, String)> {
    let mut parameters: Vec<(String, String)> = Vec::new();
    for tt in item.clone() {
        match &tt {
            TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => {
                let mut param_string = group.to_string();
                param_string.retain(|c| !c.is_whitespace());

                if param_string.contains(":") {
                    let name_start = 1;
                    let name_end = param_string[name_start..].find(':').unwrap() + 1;

                    let type_start = name_end + 1;
                    let type_end = param_string[name_start..]
                        .find(',')
                        .unwrap_or(param_string.len())
                        - 1;

                    parameters.push((
                        param_string[name_start..name_end].to_string(),
                        param_string[type_start..type_end].to_string(),
                    ));
                }
                break;
            }
            _ => (),
        }
    }
    parameters
}

fn gather_parameters_and_fix(item: &mut TokenStream) -> Vec<(String, String, String)> {
    let mut tokens = Vec::new();
    let mut parameters: Vec<(String, String, String)> = Vec::new();
    for tt in item.clone() {
        match &tt {
            TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => {
                let mut param_string = group.stream().to_string();
                param_string.retain(|c| !c.is_whitespace());

                if param_string.contains(":") {
                    let name_start = 0;
                    let name_end = param_string[name_start..].find(':').unwrap() + 1;

                    let type_start = name_end;
                    let type_end = param_string[name_start..]
                        .find(',')
                        .unwrap_or(param_string.len());

                    let original_type = param_string[type_start..type_end].to_string();
                    let new_type = format!("spirv_std::storage_class::Input<{}>", original_type);

                    param_string.replace_range(type_start..type_end, &new_type);
                    parameters.push((
                        param_string[name_start..name_end].to_string(),
                        original_type,
                        new_type,
                    ));
                    tokens.push(TokenTree::Group(Group::new(
                        Delimiter::Parenthesis,
                        TokenStream::from_str(&param_string).unwrap(),
                    )));
                } else {
                    tokens.push(tt);
                }
            }
            _ => tokens.push(tt),
        }
    }

    *item = tokens.into_iter().collect();
    parameters
}

#[proc_macro_attribute]
pub fn gpu(_: TokenStream, item: TokenStream) -> TokenStream {
    if cfg!(feature = "cpu") {
        // find original function name
        let mut func_name = "unknown".to_string();
        let mut next_is_function_name = false;
        for tt in item.clone().into_iter() {
            if next_is_function_name {
                func_name = tt.to_string();
                break;
            }
            next_is_function_name = tt.to_string() == "fn";
        }
        let cpu_func_name = format!("{}__cpu", func_name);

        // make copy of function and rename to []__cpu
        let new_cpu_func_str = item.clone().to_string().replace(&func_name, &cpu_func_name);

        // find and strip return type
        let (item, return_type) = find_and_strip_return_type(item);

        // find return expression
        if return_type.is_some() {
            let mut return_expression: Option<String> = None;

            for tt in item.clone() {
                match &tt {
                    TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => {
                        let body_str = group.to_string();
                        if let Some(start_idx) = body_str.find("return ") {
                            let start_idx = start_idx;
                            let end_idx = body_str[start_idx..].find(";").unwrap() + 1;
                            return_expression = Some(body_str[start_idx + 7..end_idx].to_string());
                        }
                    }
                    _ => (),
                }
            }
            assert!(
                return_expression.is_some(),
                "fucntions with return value must always use `return`.",
            );
        }

        // launch compute
        let mut tokens = Vec::new();
        let launch_str = format!(
            "{{ gpgpu_backend::CONTEXT.lock().unwrap().launch(\"{}\"); }}",
            func_name
        );
        for tt in item {
            match &tt {
                TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => {
                    let new_stream = TokenStream::from_str(&launch_str).unwrap();
                    tokens.push(TokenTree::Group(Group::new(Delimiter::Brace, new_stream)))
                }
                _ => tokens.push(tt),
            }
        }
        // final gpu func stream
        let gpu_stream: TokenStream = tokens.into_iter().collect();

        // combine gpu and cpu func.
        let mut combined_string = gpu_stream.to_string();
        combined_string.push_str(&new_cpu_func_str);
        TokenStream::from_str(&combined_string).unwrap()
    } else {
        // find and strip return type
        let (mut new, return_type) = find_and_strip_return_type(item);

        gather_parameters_and_fix(&mut new);

        if return_type.is_some() {
            // fixup return expression
            let mut tokens: Vec<TokenTree> = Vec::new();
            for tt in new.clone() {
                match &tt {
                    TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => {
                        let mut body_str = group.to_string();
                        if let Some(start_idx) = body_str.find("return ") {
                            let start_idx = start_idx;
                            let end_idx = body_str[start_idx..].find(";").unwrap() + 1;
                            let return_expression = body_str[start_idx + 7..end_idx].to_string();

                            // temporarily change `return foo;` into `foo`;
                            body_str.replace_range(start_idx..end_idx, &return_expression);
                            tokens.push(TokenTree::Group(Group::new(
                                Delimiter::Brace,
                                TokenStream::from_str(&body_str).unwrap(),
                            )))
                        }
                    }
                    _ => tokens.push(tt),
                }
            }
            new = tokens.into_iter().collect();

            // add output parameters
            let mut tokens: Vec<TokenTree> = Vec::new();
            for tt in new.clone() {
                match &tt {
                    TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => {
                        let mut param_tokens: VecDeque<TokenTree> =
                            group.stream().into_iter().collect();

                        param_tokens.push_front(TokenTree::Punct(Punct::new(',', Spacing::Joint)));
                        param_tokens.push_front(TokenTree::Punct(Punct::new('>', Spacing::Joint)));
                        param_tokens.push_front(TokenTree::Ident(Ident::new(
                            return_type.as_ref().unwrap().as_str(),
                            Span::call_site(),
                        )));
                        param_tokens.push_front(TokenTree::Punct(Punct::new('<', Spacing::Joint)));
                        param_tokens
                            .push_front(TokenTree::Ident(Ident::new("Output", Span::call_site())));
                        param_tokens.push_front(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                        param_tokens.push_front(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                        param_tokens.push_front(TokenTree::Ident(Ident::new(
                            "storage_class",
                            Span::call_site(),
                        )));
                        param_tokens.push_front(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                        param_tokens.push_front(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                        param_tokens.push_front(TokenTree::Ident(Ident::new(
                            "spirv_std",
                            Span::call_site(),
                        )));
                        param_tokens.push_front(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
                        param_tokens
                            .push_front(TokenTree::Ident(Ident::new("output", Span::call_site())));

                        tokens.push(TokenTree::Group(Group::new(
                            Delimiter::Parenthesis,
                            param_tokens.into_iter().collect(),
                        )));
                    }
                    _ => tokens.push(tt),
                }
            }
            new = tokens.into_iter().collect();
        }

        // strip async. TODO: Shoulbe be done without a string replace.
        let mut new_string = new.to_string();
        new_string = new_string.replace("async ", "");
        // insert entry point attribute.
        new_string.insert_str(0, "#[spirv(gl_compute)] ");
        TokenStream::from_str(&new_string).unwrap()
    }
}
