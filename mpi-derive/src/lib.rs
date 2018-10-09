#![recursion_limit = "256"]

extern crate proc_macro;
extern crate proc_macro2;

#[macro_use]
extern crate quote;
extern crate syn;

type TokenStream1 = proc_macro::TokenStream;
type TokenStream2 = proc_macro2::TokenStream;

#[proc_macro_derive(Equivalence)]
pub fn create_user_datatype(input: TokenStream1) -> TokenStream1 {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse struct");
    let result = match ast.data {
        syn::Data::Enum(_) => panic!("#[derive(Equivalence)] is not compatible with enums"),
        syn::Data::Union(_) => panic!("#[derive(Equivalence)] is not compatible with unions"),
        syn::Data::Struct(ref s) => new_for_struct(&ast, &s.fields),
    };
    result.into()
}

fn new_for_struct(ast: &syn::DeriveInput, fields: &syn::Fields) -> TokenStream2 {
    let ident = &ast.ident;

    let field_blocklengths = fields.iter().map(|_| quote!{1 as ::mpi::Count});
    let blocklengths = quote!{[#(#field_blocklengths),*]};

    let field_displacements = fields.iter().map(|field| {
        let field_ident = field.ident.as_ref().unwrap();
        quote!(
            {
                let value: #ident = unsafe { ::std::mem::uninitialized() };

                let value_loc = &value as *const _ as ::mpi::Address;

                let offset_loc = &value.#field_ident as *const _ as ::mpi::Address;

                ::std::mem::forget(value);

                offset_loc - value_loc
            }
        )
    });
    let displacements = quote!{[#(#field_displacements),*]};

    let field_datatypes = fields.iter().map(|field| {
        let ty = &field.ty;
        quote!(
            ::mpi::datatype::UncommittedDatatypeRef::from(
                <#ty as ::mpi::datatype::Equivalence>::equivalent_datatype()))
    });
    let datatypes = quote!{[#(#field_datatypes),*]};

    quote!{
        unsafe impl ::mpi::datatype::Equivalence for #ident {
            type Out = ::mpi::datatype::DatatypeRef<'static>;
            fn equivalent_datatype() -> Self::Out {
                use ::mpi::raw::AsRaw;

                thread_local!(static DATATYPE: ::mpi::datatype::DatatypeRef<'static> = {
                    let datatype =
                        ::mpi::datatype::UserDatatype::structured(
                            &#blocklengths,
                            &#displacements,
                            &#datatypes,
                        );

                    let datatype_ref =
                        unsafe { ::mpi::datatype::DatatypeRef::from_raw(datatype.as_raw()) };

                    ::std::mem::forget(datatype);

                    datatype_ref
                });

                DATATYPE.with(|datatype| datatype.clone())
            }
        }
    }
}
