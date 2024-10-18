#![recursion_limit = "256"]

type TokenStream1 = proc_macro::TokenStream;
type TokenStream2 = proc_macro2::TokenStream;

use quote::{quote, ToTokens};
use syn::{DeriveInput, Error, Expr, Fields, Type};

/// The `derive` crate feature enables the `Equivalence` derive macro, which makes it easy to
/// send structs over-the-wire without worrying about safety around padding,
/// and allowing arbitrary datatype matching between structs with the same field order but different layout.
///
/// # Example
/// ```ignore
/// use mpi_derive::Equivalence;
///
/// #[derive(Equivalence)]
/// struct MyProgramOpts {
///     name: [u8; 100],
///     num_cycles: u32,
///     material_properties: [f64; 20],
/// }
/// ```
///
/// If you use `mpi` via a re-export, you can modify the crate path using the `mpi` attribute:
/// ```ignore
/// use mpi_derive::Equivalence;
///
/// #[derive(Equivalence)]
/// #[mpi(crate = "::crate1::mpi")]
/// struct MyProgramOpts {
///     name: [u8; 100],
///     num_cycles: u32,
///     material_properties: [f64; 20],
/// }
/// ```
#[proc_macro_derive(Equivalence, attributes(mpi))]
pub fn create_user_datatype(input: TokenStream1) -> TokenStream1 {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse struct");
    let result = match ast.data {
        syn::Data::Enum(_) => panic!("#[derive(Equivalence)] is not compatible with enums"),
        syn::Data::Union(_) => panic!("#[derive(Equivalence)] is not compatible with unions"),
        syn::Data::Struct(ref s) => equivalence_for_struct(&ast, &s.fields),
    };
    result.into()
}

fn equivalence_for_tuple_field(
    mpi_crate_path: &TokenStream2,
    type_tuple: &syn::TypeTuple,
) -> TokenStream2 {
    let field_blocklengths = type_tuple.elems.iter().map(|_| 1);

    let fields = type_tuple
        .elems
        .iter()
        .enumerate()
        .map(|(i, _)| syn::Index::from(i));

    let field_datatypes = type_tuple
        .elems
        .iter()
        .map(|ty| equivalence_for_type(mpi_crate_path, ty));

    quote! {
        &#mpi_crate_path::datatype::UncommittedUserDatatype::structured(
            &[#(#field_blocklengths as #mpi_crate_path::Count),*],
            &[#(#mpi_crate_path::internal::memoffset::offset_of_tuple!(#type_tuple, #fields) as #mpi_crate_path::Address),*],
            &[#(#mpi_crate_path::datatype::UncommittedDatatypeRef::from(#field_datatypes)),*],
        )
    }
}

fn equivalence_for_array_field(
    mpi_crate_path: &TokenStream2,
    type_array: &syn::TypeArray,
) -> TokenStream2 {
    let ty = equivalence_for_type(mpi_crate_path, &type_array.elem);
    let len = &type_array.len;
    // We use the len block to ensure that len is of type `usize` and not type
    // {integer}. We know that `#len` should be of type `usize` because it is an
    // array size.
    quote! { &#mpi_crate_path::datatype::UncommittedUserDatatype::contiguous(
        {let len: usize = #len; len}.try_into().expect("rsmpi derive: Array size is to large for MPI_Datatype i32"), &#ty)
    }
}

fn equivalence_for_type(mpi_crate_path: &TokenStream2, ty: &syn::Type) -> TokenStream2 {
    match ty {
        Type::Path(ref type_path) => quote!(
                <#type_path as #mpi_crate_path::datatype::Equivalence>::equivalent_datatype()),
        Type::Tuple(ref type_tuple) => equivalence_for_tuple_field(mpi_crate_path, type_tuple),
        Type::Array(ref type_array) => equivalence_for_array_field(mpi_crate_path, type_array),
        _ => panic!("Unsupported type!"),
    }
}

fn equivalence_for_struct(ast: &syn::DeriveInput, fields: &Fields) -> TokenStream2 {
    let ident = &ast.ident;

    let field_blocklengths = fields.iter().map(|_| 1);

    let field_names = fields
        .iter()
        .enumerate()
        .map(|(i, field)| -> Box<dyn quote::ToTokens> {
            if let Some(ident) = field.ident.as_ref() {
                // named struct fields
                Box::new(ident)
            } else {
                // tuple struct fields
                Box::new(syn::Index::from(i))
            }
        });

    // parse crate path. If that fails, convert the parse error into a compile error.
    let crate_path_res = mpi_crate_path(ast);

    match crate_path_res {
        Ok(mpi_crate_path) => {
            let field_datatypes = fields
                .iter()
                .map(|field| equivalence_for_type(&mpi_crate_path, &field.ty));

            let ident_str = ident.to_string();

            // TODO and NOTE: Technically this code can race with MPI init and finalize, as can any other
            // code in rsmpi that interacts with the MPI library without taking a handle to `Universe`.
            // This requires larger attention, and so currently this is not addressed.
            quote! {
                unsafe impl #mpi_crate_path::datatype::Equivalence for #ident {
                    type Out = #mpi_crate_path::datatype::DatatypeRef<'static>;
                    fn equivalent_datatype() -> Self::Out {
                        use #mpi_crate_path::internal::once_cell::sync::Lazy;
                        use ::std::convert::TryInto;

                        static DATATYPE: Lazy<#mpi_crate_path::datatype::UserDatatype> = Lazy::new(|| {
                            #mpi_crate_path::datatype::internal::check_derive_equivalence_universe_state(#ident_str);

                            #mpi_crate_path::datatype::UserDatatype::structured::<
                                #mpi_crate_path::datatype::UncommittedDatatypeRef,
                            >(
                                &[#(#field_blocklengths as #mpi_crate_path::Count),*],
                                &[#(#mpi_crate_path::internal::memoffset::offset_of!(#ident, #field_names) as #mpi_crate_path::Address),*],
                                &[#(#mpi_crate_path::datatype::UncommittedDatatypeRef::from(#field_datatypes)),*],
                            )
                        });

                        DATATYPE.as_ref()
                    }
                }
            }
        }
        Err(e) => e.into_compile_error(),
    }
}

fn mpi_crate_path(input: &DeriveInput) -> syn::Result<TokenStream2> {
    const MPI_CRATE_PATH_ATTR: &str = "mpi";
    const META_PATH: &str = "crate";

    let crate_path_attrs: Vec<_> = input
        .attrs
        .iter()
        .filter(|input| input.path().is_ident(MPI_CRATE_PATH_ATTR))
        .collect();

    if crate_path_attrs.is_empty() {
        Ok(quote! {::mpi})
    } else if crate_path_attrs.len() > 1 {
        Err(Error::new_spanned(
            input,
            "Only one `mpi` attribute is allowed",
        ))
    } else {
        let crate_path_attr = crate_path_attrs[0];

        let mut crate_path = None;

        crate_path_attr
            .parse_nested_meta(|meta| {
                if !meta.path.is_ident(META_PATH) {
                    return Err(Error::new_spanned(
                        &meta.path,
                        format!(
                            "unexpected attribute `{}`. Expected `crate`",
                            meta.path.to_token_stream()
                        ),
                    ));
                }

                let expr: Expr = meta.value()?.parse()?;
                let mut value = &expr;

                // unpack (unnecessary) parentheses
                while let Expr::Group(e) = value {
                    value = &e.expr;
                }

                // expect a string literal that parses to a crate path
                if let Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(lit),
                    ..
                }) = value
                {
                    let suffix = lit.suffix();
                    if !suffix.is_empty() {
                        return Err(Error::new_spanned(
                            lit,
                            format!("Unexpected suffix `{}` on string literal", suffix),
                        ));
                    }

                    crate_path = match lit.parse() {
                        Ok(path) => {
                            if crate_path.is_some() {
                                return Err(Error::new_spanned(
                                    meta.path,
                                    "Duplicate `crate` attribute",
                                ));
                            }
                            Some(path)
                        }
                        Err(_) => {
                            return Err(Error::new_spanned(
                                lit,
                                format!("Failed to parse path: {:?}", lit.value()),
                            ))
                        }
                    };

                    Ok(())
                } else {
                    Err(Error::new_spanned(
                        value,
                        "Expected string literal containing crate path",
                    ))
                }
            })
            .map(|_| crate_path.unwrap())
    }
}
